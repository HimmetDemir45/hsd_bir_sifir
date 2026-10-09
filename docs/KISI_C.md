# Kişi C: Frontend tasarımı görev dosyası

> **Claude Code için:** Önce `CLAUDE.md`'yi, sonra bu dosyayı baştan sona oku. Sen **Kişi C**'nin asistanısın.
> Görevin dashboard'un **görünümü**: `web/index.html` iskeleti, `web/style.css` ve görseller. Davranış (JavaScript)
> ve sunucu **Kişi B**'de. Sadece aşağıdaki "Sorumluluk alanı"ndaki dosyalara dokun. Her adımdan sonra kullanıcıya
> nasıl test edeceğini kısa yaz.

## 0. Kurulum (bir kez)

Windows, Python 3.11 gerekli (`py -0` ile kontrol et; yoksa python.org'dan 3.11 kur).

```bash
git clone https://github.com/HimmetDemir45/hsd_bir_sifir.git
cd hsd_bir_sifir
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
git checkout -b tasarim
```

Model indirmen **gerekmez**: tasarım için demo modu yeterli (kamera ve model kullanmaz, sahte olay üretir).

```bash
python main.py --demo --fresh --speed 4
```
Tarayıcıda `http://localhost:8000`. Yaklaşık 15 saniyede sarı → turuncu → kırmızı uyarılar akar, kırmızıda tam ekran
modal açılır. `--fresh` eski kayıtları siler; `--speed 4` senaryoyu 4 kat hızlandırır. CSS/HTML değiştirince sayfayı
yenilemen yeterli (sunucu önbelleği kapalı, yeniden başlatma gerekmez).

## 1. Sorumluluk alanı

| Kişi C (sen) | Kişi B | Dokunma |
|---|---|---|
| `web/index.html` (iskelet, metinler), `web/style.css`, `web/assets/` (yeni görseller, yerel fontlar, ikonlar) | `web/app.js`, `api/`, `fusion/`, `storage/`, `notify/`, `main.py` | Kişi A: `sources/`, `detectors/`, `privacy/`, `tools/`, `models/` |

- **JavaScript değişikliği gerekirse** (yeni bir veri göstermek, yeni buton): kendin yazma, **B'ye ne istediğini söyle**.
  B ve sen aynı dosyayı değiştirirseniz sürekli çakışırsınız.
- `web/floorplan.png` (kat planı) değişecekse önce B ile konuş: bölgelerin poligon koordinatları `config.yaml` →
  `zones` altında ve 760x500 tuval koordinatlarında. Yeni planda bölgeler başka yerdeyse koordinatlar da değişmeli.
- Ortak dosyalar (`README.md`, `config.yaml`, `requirements.txt`) gerekmedikçe değişmez; değişecekse önce haber ver.

## 2. JavaScript ile sözleşme (BUNLARI KORU)

`web/app.js` sayfadaki elemanları **id** ile bulur ve **sınıf** atar. Bunlar değişirse dashboard sessizce bozulur.

### Silinmemesi / yeniden adlandırılmaması gereken id'ler
| id | Ne |
|---|---|
| `conn` | bağlantı rozeti (JS `conn on` / `conn off` sınıfı ve metni atar) |
| `soundBtn` | alarm sesi aç/kapa düğmesi (JS `sound on` / `sound off` atar). Tarayıcılar sesi ancak bir tıklamadan sonra çalar; bu düğme o tıklama, kaldırma |
| `cam` | canlı görüntü `<img src="/video/cam1">` (MJPEG). Bir `<img>` olarak kalmalı |
| `count`, `alerts` | uyarı sayısı ve uyarı listesi `<ul>` |
| `plan` | kat planı `<canvas width="760" height="500">` (ısı haritası JS ile çizilir; CSS ile ölçekleyebilirsin, width/height özniteliklerini değiştirme) |
| `weekly` | haftalık özet tablosu (JS `#weekly tbody`'yi doldurur) |
| `redModal`, `redZone`, `redSnap`, `redReasons`, `redClip`, `redActions`, `confirmBtn`, `dismissBtn`, `redResult`, `closeBtn` | kırmızı uyarı modalı ve parçaları |
| `alarm` | `<audio src="/alarm.wav" loop>` |

`hidden` özniteliğini JS açıp kapatır (modal, sonuç ekranı, klip bağlantısı). `style.css`'teki
`[hidden] { display: none !important; }` kuralını **silme**: yoksa `display: flex` gibi bir kural gizli elemanı gösterir.

### JS'in ürettiği uyarı kartı (bunu CSS ile tasarla)
```html
<li class="alert yellow|orange|red [fresh]">        <!-- fresh: yeni gelen, kısa süre vurgulanabilir -->
  <div class="top"><span class="lvl">Sarı · Kantin</span><span>16:41:53</span></div>
  <ul><li>Kavga şüphesi</li><li>...</li></ul>        <!-- sebepler (birden fazla olabilir) -->
  <a class="clip-link">▶ Klibi izle</a>              <!-- sadece turuncu/kırmızıda, birkaç sn sonra gelir -->
  <div class="status">Bekliyor | Onaylandı | Yanlış alarm</div>
</li>
```
Diğer JS sınıfları: `#weekly td.empty` ("Henüz uyarı yok"), tablo başlıklarında `th.y`, `th.o`, `th.r`.

### Veri (göz atmak için, değiştirmek için değil)
- `GET /api/alerts`: uyarı listesi; `GET /api/summary/weekly`, `GET /api/heatmap`, `GET /api/zones`
- WebSocket `/ws`: `{"type": "alert", "alert": {...}}`
- Alert alanları: `level` (`yellow`/`orange`/`red`), `reasons` (Türkçe metin listesi), `zone_id`, `created_at`,
  `status` (`pending`/`confirmed`/`dismissed`), `snapshot`, `clip`

## 3. Değişmez ilkeler (CLAUDE.md'den, bunlara aykırı tasarım yapma)
- **İnternetsiz çalışmalı:** CDN, Google Fonts, uzaktan ikon/kütüphane **yok**. Font kullanacaksan dosyasını
  `web/assets/` içine koy ve `@font-face` ile yükle (lisansı serbest olsun). Framework yok (Tailwind, Bootstrap, React yok).
- **112 araması simülasyondur:** modaldeki "112'yi ara (simülasyon)" ve "112 arandı (simülasyon)" ifadelerindeki
  "simülasyon" kelimesi kalmalı. Onay butonu bir insanın bilinçli tıklaması olmalı (otomatik/yanıltıcı tasarım yok).
- **Gizlilik:** görüntüler sunucudan yüzleri bulanık gelir; tasarımda yüz gösteren örnek/stok görsel kullanma.
- Kırmızı modal, en dikkat çekici öğe olmalı (uzaktan, projeksiyonda okunur). Renk tek başına anlam taşımasın:
  seviye metni ("KIRMIZI", "Turuncu") görünür kalsın.
- Sunum projeksiyonda / laptopta: 1366x768 ve 1920x1080'de düzgün görünmeli; canlı video dikey de olabilir
  (test videosu 1080x1920).

## 4. Önerilen işler (sırayla, her biri ayrı commit)
1. **Renk/tema ve tipografi:** `style.css` başındaki `:root` değişkenleri. Sarı/turuncu/kırmızı ayırt edilebilir
   ve koyu arka planda okunur olsun.
2. **Üst bar:** logo/ad, bağlantı rozeti, alarm düğmesi; okul adı/tarih eklenebilir (statik metin index.html'de).
3. **Uyarı kartları:** seviyeye göre sol kenar/ikon, yeni gelen (`.fresh`) için kısa animasyon, okunur zaman.
4. **Kırmızı modal:** tam ekran, büyük başlık, snapshot büyük, iki buton net ayrışsın (Onayla vs Yanlış alarm).
5. **Alt bölüm:** kat planı ve haftalık özet tablosu.
6. **(İsteğe bağlı) Kat planı görseli:** daha gerçekçi bir okul planı. Önce B ile poligon koordinatlarını konuşun.

Test: `python main.py --demo --fresh --speed 4` ile senaryonun tamamını izle; kırmızı modalda **Onayla** ve
**Yanlış alarm** ikisini de dene (ikinci deneme için sunucuyu `--fresh` ile yeniden başlat), "Alarm sesi" düğmesine bas.
Tarayıcı penceresini daraltıp genişleterek düzeni kontrol et.

## 5. Git akışı
- Kendi dalında çalış: `tasarim` (A: `goruntu-ses`, B: `fuzyon-web`).
- Küçük ve sık commit; sadece kendi dosyalarını ekle:
  ```bash
  git add web/index.html web/style.css web/assets/
  git commit -m "web: uyarı kartı tasarımı"
  git push -u origin tasarim      # ilk seferde -u, sonra sadece git push
  ```
  `git add .` kullanma. Satır sonları `.gitattributes` ile otomatik ayarlanır.
- Main'deki yenilikleri almak (B'nin yeni JS'i vb.): `git fetch origin` ve `git merge origin/main`.
- Main'e birleştirme: kilometre taşlarında, **A ve B'ye haber vererek, sırayla**:
  ```bash
  git fetch origin
  git merge origin/main          # önce main'i al, çakışma varsa çöz, demo ile test et
  git push
  git checkout main
  git pull
  git merge tasarim
  git push
  git checkout tasarim
  ```
- **06.30 code freeze**: sonrasında sadece kritik düzeltme.
