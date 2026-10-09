"""Yüz bulanıklaştırma (gizlilik ilkesi: yayında ve kayıtlarda yüzler görünmez).

İLKE: Yüz TANIMA yok. Yüzün sadece yeri bulunur ve o bölge pikselleştirilir; kimlik çıkarılmaz,
yüz özelliği saklanmaz.

İki kaynağın birleşimi bulanıklaştırılır (kaçırmak, fazladan bulanıklaştırmaktan kötü):
1) Poz keypoint'lerinden kafa bölgesi: burun/göz/kulak; görünmüyorsa omuzların üstü; o da yoksa
   kişi kutusunun üst kısmı. Güvenlik kamerası açılarında (yandan, tepeden) yüz dedektöründen iyi.
2) Yüz dedektörü: OpenCV YuNet (models/face_detection_yunet_2023mar.onnx, MIT);
   model yoksa OpenCV ile gelen Haar cascade. Pozun kaçırdığı (ör. yarım görünen) yüzler için.

Kullanım (main.py kamera döngüsü, overlay'ler çizildikten sonra, yayın/snapshot/klipten önce):
    from privacy.blur import blur_faces
    frame = blur_faces(frame, pose.privacy_persons)   # düşük güvenli kişiler dahil (opsiyonel)

Tek başına deneme:
    python -m privacy.blur                                  # webcam, bulanık görüntü penceresi
    python -m privacy.blur --source test_media/Kavga.mp4
    python -m privacy.blur --image foto.jpg --save cikti.jpg
"""
from __future__ import annotations

import argparse
import threading
import time
from typing import Any, Iterable

import cv2
import numpy as np

from core.config import load_config, resolve_path

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 0, 1, 2, 3, 4, 5, 6
HEAD_KPS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class FaceBlurrer:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["privacy"]
        self.enabled: bool = bool(self.cfg.get("blur_faces", True))
        self.kp_conf: float = cfg["pose"]["kp_conf"]
        self.reliable_conf: float = cfg["pose"]["person_conf"]  # bunun altındaki kişide keypoint'ler kayabiliyor
        self._lock = threading.Lock()
        self._recent: list[tuple[float, Box]] = []  # (zaman, kutu): zamansal hafıza
        self._yunet = None
        self._haar = None
        model = resolve_path(self.cfg["face_model"])
        if model.is_file():
            self._yunet = cv2.FaceDetectorYN.create(str(model), "", (320, 320),
                                                    score_threshold=self.cfg["face_conf"])
            self.detector_name = "yunet"
            return
        print(f"[privacy] UYARI: {model} yok -> yüz dedektörü zayıf yedeğe düşüyor. İndirmek için: "
              f"python -m tools.modelleri_indir")
        # Haar yedeği: OpenCV 5.0 pip paketinde cascade dosyaları yok -> empty() kontrolü şart
        haar_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
        haar = cv2.CascadeClassifier(haar_dir + "haarcascade_frontalface_default.xml") if haar_dir else None
        if haar is not None and not haar.empty():
            self._haar = haar
            self.detector_name = "haar"
        else:
            self.detector_name = "yok (sadece poz)"
            print("[privacy] UYARI: Haar cascade da bulunamadı -> sadece poz kafa bölgeleri bulanıklaştırılıyor. "
                  "Yüz kaçırma riski yüksek; YuNet modelini indirin.")

    # ---------- kafa / yüz kutuları ----------
    def head_boxes(self, person: Any) -> list[Box]:
        """Kişi başına kafa kutuları: keypoint kutusu + ayaktaysa kişi kutusunun üst çeyreği.
        Düşük güvenli kişilerde keypoint'ler kayabiliyor (Kavga.mp4 9. sn: kutu göğse düştü);
        kişi kutusu ise genelde doğru, üst bandı da kapatmak yüzü kaçırmayı önler."""
        boxes: list[Box] = []
        kp_box = self.head_box(person)
        if kp_box is not None:
            boxes.append(kp_box)
        x1, y1, x2, y2 = person.box
        # Üst bant sadece keypoint'lere güvenilemeyen kişide (düşük güven ya da kafa noktası yok):
        # yüksek güvenli kişide gereksiz yere geniş alan kapatıp olayı izlenemez yapıyordu.
        unreliable = kp_box is None or person.conf < self.reliable_conf
        if unreliable and (y2 - y1) >= (x2 - x1):
            boxes.append((x1, y1, x2, y1 + (y2 - y1) * self.cfg["top_band"]))
        return boxes

    def head_box(self, person: Any) -> Box | None:
        """Poz keypoint'lerinden yüz kutusu: genişlik kulaktan kulağa (yoksa göz arası / profil),
        yükseklik ~1.3 x genişlik, burun merkezli ve alına doğru kaydırılmış (boyna inmez).
        Omuz genişliği sadece hiç kafa noktası yoksa kullanılır (yakın kamerada kutuyu boyna kadar büyütüyordu)."""
        kps, kpc = person.kps, person.kpc
        x1, y1, x2, y2 = person.box
        ok = kpc >= self.kp_conf

        def dist(a: int, b: int) -> float:
            return float(np.linalg.norm(kps[a] - kps[b]))

        ears = [i for i in (L_EAR, R_EAR) if ok[i]]
        eyes = [i for i in (L_EYE, R_EYE) if ok[i]]
        head_pts = [i for i in HEAD_KPS if ok[i]]
        shoulders = ok[L_SH] and ok[R_SH]

        # yüz genişliği: mevcut tahminlerin en büyüğü (yan dönen kişide göz arası kısalıyor, tek başına küçük kalır)
        estimates = []
        if len(ears) == 2:
            estimates.append(dist(L_EAR, R_EAR) * 1.15)
        if len(eyes) == 2:
            estimates.append(dist(L_EYE, R_EYE) * 2.4)
        for ear in ears:  # profil: kulak ile göz/burun arası ~ yarım yüz
            if eyes or ok[NOSE]:
                estimates.append(dist(ear, eyes[0] if eyes else NOSE) * 1.6)
        if shoulders:  # alt sınır: kafa genişliği omuzun ~%35-50'si (yalnız başına kullanılırsa %50)
            estimates.append(dist(L_SH, R_SH) * (0.35 if estimates else 0.5))
        w = max(estimates) if estimates else 0.1 * max(x2 - x1, y2 - y1)

        # yüz merkezi
        if ok[NOSE]:
            cx, cy = kps[NOSE]
        elif head_pts:
            cx, cy = kps[head_pts].mean(axis=0)
        elif shoulders:
            mid = (kps[L_SH] + kps[R_SH]) / 2
            cx, cy = mid[0], mid[1] - w * 1.2
        elif (y2 - y1) >= (x2 - x1):  # ayakta, hiç keypoint yok: kutunun üst kısmı
            return (x1, y1, x2, y1 + (y2 - y1) * self.cfg["top_band"])
        else:
            return None

        w = max(w, 8.0) * self.cfg["head_scale"]
        h = w * 1.3
        return (cx - w / 2, cy - 0.6 * h, cx + w / 2, cy + 0.4 * h)

    def face_boxes(self, frame: np.ndarray) -> list[Box]:
        h, w = frame.shape[:2]
        scale = min(1.0, self.cfg["face_max_side"] / max(h, w))
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
        boxes: list[Box] = []
        if self._yunet is not None:
            self._yunet.setInputSize((small.shape[1], small.shape[0]))
            _, faces = self._yunet.detect(small)
            if faces is not None:
                for fx, fy, fw, fh in faces[:, :4]:
                    boxes.append((fx / scale, fy / scale, (fx + fw) / scale, (fy + fh) / scale))
        elif self._haar is not None:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            for fx, fy, fw, fh in self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4):
                boxes.append((fx / scale, fy / scale, (fx + fw) / scale, (fy + fh) / scale))
        return boxes

    # ---------- bulanıklaştırma ----------
    def _pixelate(self, frame: np.ndarray, box: Box) -> None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        pad = self.cfg["pad"] * min(x2 - x1, y2 - y1)  # kısa kenara göre: geniş bantlar devleşmesin
        x1, y1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        x2, y2 = int(min(w, x2 + pad)), int(min(h, y2 + pad))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return
        roi = frame[y1:y2, x1:x2]
        blocks = self.cfg["pixel_blocks"]
        bw = max(1, min(blocks, (x2 - x1) // 2))
        bh = max(1, min(blocks, (y2 - y1) // 2))
        small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_AREA)
        frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    def boxes(self, frame: np.ndarray, tracks: Iterable[Any] | None = None) -> list[Box]:
        result: list[Box] = []
        for t in tracks or ():
            result += self.head_boxes(getattr(t, "person", t))  # Track veya Person kabul eder
        result += self.face_boxes(frame)
        return result

    def blur(self, frame: np.ndarray, tracks: Iterable[Any] | None = None,
             now: float | None = None) -> np.ndarray:
        """Kareyi yerinde bulanıklaştırır ve döndürür.
        Son persist_s saniyede bulunan bölgeler de bulanıklaştırılır: kişi hareket bulanıklığı yüzünden
        1-2 kare algılanmazsa yüzü açığa çıkmasın (Kavga.mp4 7.5-7.7. sn'de bu oluyordu)."""
        if not self.enabled:
            return frame
        now = time.time() if now is None else now
        with self._lock:
            current = self.boxes(frame, tracks)
            keep = self.cfg["persist_s"]
            # Sadece bu karede karşılığı olmayan (kaybolan kişinin) eski kutuları tut; görünen kişinin
            # eski kutularını biriktirmek hareket izini de bulanıklaştırıp sahneyi okunmaz yapıyordu.
            lost = [(t, b) for t, b in self._recent
                    if now - t <= keep and all(_iou(b, c) < 0.3 for c in current)]
            self._recent = [(now, b) for b in current] + lost
            for _, box in self._recent:
                self._pixelate(frame, box)
        return frame


_blurrer: FaceBlurrer | None = None
_init_lock = threading.Lock()


def get_blurrer(cfg: dict | None = None) -> FaceBlurrer:
    global _blurrer
    with _init_lock:
        if _blurrer is None:
            _blurrer = FaceBlurrer(cfg or load_config())
        return _blurrer


def blur_faces(frame: np.ndarray, tracks: Iterable[Any] | None = None,
               now: float | None = None) -> np.ndarray:
    """Yüzleri pikselleştirir (yerinde) ve kareyi döndürür.
    tracks: kişiler (PoseDetector.privacy_persons önerilir), now: kare zamanı (yoksa time.time())."""
    return get_blurrer().blur(frame, tracks, now)


def main() -> None:
    import time

    from detectors.pose import PoseDetector
    from sources.camera import Camera

    cfg = load_config()
    parser = argparse.ArgumentParser(description="Yüz bulanıklaştırmayı tek başına dene.")
    parser.add_argument("--source", default=str(cfg["cameras"][0]["source"]), help="0 = webcam veya video dosyası")
    parser.add_argument("--image", default=None, help="tek fotoğraf")
    parser.add_argument("--save", default=None, help="--image ile: sonucu bu dosyaya yaz")
    parser.add_argument("--no-pose", action="store_true", help="sadece yüz dedektörü (karşılaştırma için)")
    args = parser.parse_args()

    blurrer = get_blurrer(cfg)
    pose = None if args.no_pose else PoseDetector(cfg, "cam1", "test")
    if pose is not None:
        pose.cfg = {**pose.cfg, "every_n_frames": 1}
    print(f"[privacy] yüz dedektörü={blurrer.detector_name} poz={'kapalı' if pose is None else 'açık'}")

    if args.image:
        img = cv2.imread(str(resolve_path(args.image)))
        persons = None
        if pose:
            pose.process(img)
            persons = pose.privacy_persons  # düşük güvenli kişiler dahil
        n = len(blurrer.boxes(img, persons))
        blurrer.blur(img, persons)
        out = args.save or "blur_sonuc.jpg"
        cv2.imwrite(out, img)
        print(f"[privacy] {n} bölge bulanıklaştırıldı -> {out}")
        return

    camera = Camera(args.source)
    try:
        while (frame := camera.read()) is not None:
            t0 = time.time()
            persons = None
            if pose:
                pose.process(frame, camera.last_ts)
                persons = pose.privacy_persons
            blurrer.blur(frame, persons)
            cv2.putText(frame, f"{(time.time() - t0) * 1000:.0f} ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("OkulKalkan - privacy", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
