"""web/floorplan.png üretir (760x500). Bölge poligonları config.yaml ile AYNI kalır:
kantin (50,50)-(300,200), koridor_1 (320,50)-(700,120), spor_salonu (50,220)-(400,450), bahce (420,220)-(700,450).
Bölge adı ve olay sayısını app.js her bölgenin ortasına yazar: o alanlar boş bırakılır.
Kullanım (proje kökünden): python web/assets/draw_floorplan.py web/floorplan.png
Gerekli: Pillow (matplotlib/ultralytics ile zaten gelir), Windows Bahnschrift fontu.
"""
import math
import sys

from PIL import Image, ImageDraw, ImageFont

S = 3                      # süper örnekleme (kenar yumuşatma için 3x çiz, sonra küçült)
W, H = 760, 500
PAPER = (245, 243, 238)    # style.css .plan-frame ile aynı
WALL = (70, 72, 76)
INNER = (110, 112, 116)
FURN = (200, 195, 185)
FURN_FILL = (233, 230, 223)
LABEL = (128, 123, 114)
GRASS = (226, 234, 216)
TREE = (196, 214, 182)
TREE_LINE = (150, 172, 138)
PATH = (234, 228, 214)

FONT_PATH = "C:/Windows/Fonts/bahnschrift.ttf"


def font(size, weight=400):
    f = ImageFont.truetype(FONT_PATH, size * S)
    try:
        f.set_variation_by_axes([weight, 100])   # wght, wdth
    except Exception:
        pass
    return f


img = Image.new("RGB", (W * S, H * S), PAPER)
d = ImageDraw.Draw(img)


def P(*xy):
    return [v * S for v in xy]


def line(x1, y1, x2, y2, w, c):
    d.line(P(x1, y1, x2, y2), fill=c, width=round(w * S))
    r = w * S / 2   # yuvarlak uçlar
    for x, y in ((x1, y1), (x2, y2)):
        d.ellipse([x * S - r, y * S - r, x * S + r, y * S + r], fill=c)


def wall(points, w=4, c=WALL):
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        line(x1, y1, x2, y2, w, c)


def rect(x1, y1, x2, y2, outline=FURN, fill=FURN_FILL, w=1.2, r=0):
    if r:
        d.rounded_rectangle(P(x1, y1, x2, y2), radius=r * S, outline=outline, fill=fill, width=round(w * S))
    else:
        d.rectangle(P(x1, y1, x2, y2), outline=outline, fill=fill, width=round(w * S))


def circle(cx, cy, r, outline=FURN, fill=None, w=1.2):
    d.ellipse(P(cx - r, cy - r, cx + r, cy + r), outline=outline, fill=fill, width=round(w * S))


def arc(cx, cy, r, a0, a1, c=INNER, w=1):
    d.arc(P(cx - r, cy - r, cx + r, cy + r), a0, a1, fill=c, width=round(w * S))


def text(x, y, s, size=11, c=LABEL, anchor="mm", weight=400):
    d.text((x * S, y * S), s, font=font(size, weight), fill=c, anchor=anchor)


def door_h(x, y, width=22, swing=1):
    """Yatay duvarda kapı: duvarda boşluk + açılma yayı. swing=+1 aşağı, -1 yukarı açılır."""
    d.rectangle(P(x, y - 4, x + width, y + 4), fill=PAPER)
    line(x, y, x, y + swing * width, 1, INNER)
    if swing > 0:
        arc(x, y, width, 0, 90)
    else:
        arc(x, y, width, 270, 360)


def door_v(x, y, width=22, swing=1):
    """Dikey duvarda kapı. swing=+1 sağa, -1 sola açılır."""
    d.rectangle(P(x - 4, y, x + 4, y + width), fill=PAPER)
    line(x, y, x + swing * width, y, 1, INNER)
    if swing > 0:
        arc(x, y, width, 0, 90)
    else:
        arc(x, y, width, 90, 180)


# ---------------- Bahçe (dış alan, binanın dışında) ----------------
d.rectangle(P(414, 216, 706, 456), fill=GRASS)
# yürüyüş yolu: sınıf koridorundaki bahçe kapısından aşağı, sonra sağa
d.rectangle(P(548, 214, 572, 456), fill=PATH)
d.rectangle(P(414, 400, 706, 418), fill=PATH)
# ağaçlar (köşelerde; ortadaki etiket alanı boş)
for cx, cy, r in [(445, 250, 17), (480, 240, 12), (672, 252, 18), (640, 236, 11),
                  (448, 440, 13), (676, 440, 14), (640, 444, 9), (500, 446, 9)]:
    circle(cx, cy, r, outline=TREE_LINE, fill=TREE, w=1.2)
    circle(cx, cy, 2, outline=TREE_LINE, fill=TREE_LINE, w=1)
# banklar
rect(600, 386, 640, 393, outline=FURN, fill=FURN_FILL)
rect(470, 386, 510, 393, outline=FURN, fill=FURN_FILL)
# çit (kesikli)
def dashed(x1, y1, x2, y2, dash=6, gap=4):
    length = math.hypot(x2 - x1, y2 - y1)
    n = int(length // (dash + gap)) + 1
    for i in range(n):
        a = i * (dash + gap) / length
        b = min(1, (i * (dash + gap) + dash) / length)
        line(x1 + (x2 - x1) * a, y1 + (y2 - y1) * a, x1 + (x2 - x1) * b, y1 + (y2 - y1) * b, 1.4, TREE_LINE)
dashed(414, 456, 706, 456)
dashed(706, 216, 706, 456)

# ---------------- Kantin (50,50)-(300,200) ----------------
# büfe tezgâhı sol duvar boyunca
rect(56, 62, 72, 188, r=2)
text(64, 125, "BÜFE", 8, anchor="mm")   # dikey yazı yerine kısa etiket
# masalar: köşelerde, merkez (175,125) boş
for tx, ty in [(100, 72), (150, 72), (210, 72), (260, 72), (100, 170), (150, 170), (210, 170), (260, 170)]:
    rect(tx - 13, ty - 7, tx + 13, ty + 7, r=2)
    for sx in (-7, 7):
        circle(tx + sx, ty - 11, 3)
        circle(tx + sx, ty + 11, 3)

# ---------------- 1. Kat koridor (320,50)-(700,120) ----------------
# dolaplar üst duvar boyunca
for i in range(18):
    x = 330 + i * 20
    rect(x, 54, x + 17, 62, r=1)

# ---------------- Sınıf şeridi (310..710, 130..210): bölge değil, etiketli ----------------
rooms = [(310, 410, "Sınıf 1-A"), (410, 510, "Sınıf 1-B"), (510, 600, "Rehberlik"), (600, 650, "Müdür Yrd."), (650, 710, "WC")]
for x1, x2, name in rooms:
    text((x1 + x2) / 2, 172, name, 10 if len(name) < 10 else 9)
for x in (410, 510, 600, 650):
    wall([(x, 130), (x, 210)], 2.5, INNER)
# sıralar (sınıflarda)
for x1, x2, _ in rooms[:2]:
    for r_ in range(2):          # kapı yayından (x1+30..x1+52) uzakta, sağ yarıda
        for c_ in range(2):
            rect(x1 + 58 + c_ * 20, 138 + r_ * 12, x1 + 74 + c_ * 20, 144 + r_ * 12, r=1)

# ---------------- Spor salonu (50,220)-(400,450) ----------------
COURT = (214, 208, 196)
cx0, cy0, cx1, cy1 = 72, 240, 378, 430
d.rectangle(P(cx0, cy0, cx1, cy1), outline=COURT, width=round(1.4 * S))
mid = (cx0 + cx1) / 2
line(mid, cy0, mid, 312, 1.4, COURT)        # orta çizgi: etiket alanında (merkez ~225,340) kesik
line(mid, 368, mid, cy1, 1.4, COURT)
for side in (0, 1):                          # boyalı alanlar + potalar
    x_edge = cx0 if side == 0 else cx1
    k = 1 if side == 0 else -1
    d.rectangle(P(min(x_edge, x_edge + k * 50), 305, max(x_edge, x_edge + k * 50), 375), outline=COURT, width=round(1.4 * S))
    arc(x_edge + k * 50, 340, 22, 270, 90) if side == 0 else arc(x_edge + k * 50, 340, 22, 90, 270)
    circle(x_edge + k * 12, 340, 4, outline=COURT, w=1.4)
# tribün (alt duvar boyunca)
for i in range(3):
    line(80, 438 + i * 4, 370, 438 + i * 4, 1, FURN)

# ---------------- Duvarlar ----------------
# bina dış duvarı (L biçimi; bahçe dışarıda)
wall([(44, 44), (710, 44), (710, 210), (410, 210), (410, 456), (44, 456), (44, 44)], 6)
# iç duvarlar
wall([(310, 44), (310, 210)], 4, INNER)          # kantin | koridor+sınıflar
wall([(44, 210), (310, 210)], 4, INNER)          # kantin | spor salonu
wall([(310, 126), (710, 126)], 4, INNER)         # koridor | sınıflar
wall([(310, 210), (410, 210)], 4, INNER)         # sınıf 1-A | spor salonu

# kapılar
door_v(310, 74, 22, swing=-1)                    # kantin ↔ koridor
door_h(164, 210, 24, swing=1)                    # kantin ↔ spor salonu
for x in (340, 440, 530, 615, 665):              # koridor ↔ odalar
    door_h(x, 126, 18 if x > 600 else 22, swing=1)
door_h(549, 210, 22, swing=1)                    # sınıflar ↔ bahçe (yolun başı)
door_v(410, 404, 22, swing=1)                    # spor salonu ↔ bahçe
# ana giriş: koridorun doğu ucu
d.rectangle(P(706, 66, 714, 104), fill=PAPER)
line(710, 66, 710, 104, 1, INNER)

# ---------------- Giriş oku, başlık, kuzey, ölçek ----------------
def arrow(x1, y, x2, c=WALL):
    line(x1, y, x2, y, 2, c)
    d.polygon(P(x2, y, x2 + 7, y - 5, x2 + 7, y + 5), fill=c)
arrow(752, 85, 718)
text(736, 104, "GİRİŞ", 9, c=WALL, weight=600)

text(44, 482, "ZEMİN KAT", 12, c=WALL, anchor="lm", weight=700)
text(122, 482, "Okul yerleşim planı · temsili", 10, anchor="lm")

# kuzey oku
nx, ny = 735, 470
d.polygon(P(nx, ny - 14, nx + 6, ny + 4, nx, ny, nx - 6, ny + 4), fill=WALL)
text(nx, ny + 14, "K", 9, c=WALL, weight=600)
# ölçek çubuğu (temsili 10 m)
sx, sy = 600, 484
line(sx, sy, sx + 60, sy, 2, WALL)
for t in (0, 30, 60):
    line(sx + t, sy - 4, sx + t, sy + 1, 1.4, WALL)
text(sx + 30, sy - 11, "10 m", 9)

out = img.resize((W, H), Image.LANCZOS)
out.save(sys.argv[1], optimize=True)
print("yazıldı:", sys.argv[1], out.size)
