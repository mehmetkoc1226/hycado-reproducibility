# HyCADO 28 — YOLO-800 + YOLO-1280 füzyon kontrolü

## Çalıştırma
1. ZIP'i tamamen çıkartın; ZIP içinden CMD çalıştırmayın.
2. Çıkan klasörde `28_YOLO800_1280_FUZYON.cmd` dosyasına çift tıklayın.
3. Tamamlandığında **aynı CMD'nin yanındaki** `28_cift_cozunurluk_fuzyon_rapor.zip` dosyasını gönderin.

Dosyaları ana proje klasörüne kopyalamanız gerekmez. Proje varsayılanı `D:\HyCADO_Temiz_Baslangic` olarak ayarlıdır. Farklıysa CMD'yi terminalde ilk argüman olarak proje yolu ile çalıştırın. Test-dev annotation klasörü taşındıysa ikinci argüman `annotations` klasörünü içeren veri kümesi kökü olmalıdır:

```bat
28_YOLO800_1280_FUZYON.cmd "D:\HyCADO_Temiz_Baslangic" "D:\Veri\VisDrone2019-DET-test-dev"
```

## Ne yapıyor?
- Mevcut 1610 test-dev görüntüsünün kayıtlı YOLO-800 ve YOLO-1280 tahminlerini birleştirir. Görüntülerin kendisini, model ağırlıklarını veya GPU'yu açmaz.
- Skorlar aynı kalır (iki taraf da ×1). Native xyxy koordinatlar tekrar ölçeklenmez. Mevcut sıralama ve sınıf bazlı NMS (IoU >0.7, en fazla1000 çıktı) kullanılır.
- Füzyon tahminlerini sabitler; aynı ignore işleme ve pycocotools2.0.11 ile COCO AP, AP50/75, küçük/orta/büyük AP ve10 sınıf AP hesaplar. MaxDets100 ana,500 duyarlılık sonucudur.
- YOLO-1280 ve HyCADO-1280 sonuçlarını da yeniden hesaplayıp tamamlanmış22 sonuçlarıyla eşleşmelerini şart koşar. SAHI-640'ın tamamlanmış24 sonuçlarını yeniden çıkarım yapmadan kullanır.
- Füzyon için1000 eşleştirilmiş proxy-cluster bootstrap çekilişi hesaplar; önceki23/24 çekilişleriyle aynı grupları/tohumu kullanır. Ana fark HyCADO-1280 eksi yeni füzyondur. Füzyon eksi YOLO-1280 ve füzyon eksi SAHI-640 da raporlanır.
- Sonucun yönüne bakılmadan tüm karşılaştırmalar raporlanır. Bu kontrol, önceden incelenmiş test-dev üzerinde keşifseldir; bağımsız onaylayıcı test veya eşit hesap bütçesi deneyi değildir.

## Ortam ve kaynaklar
Windows64-bit Python3.11; mevcut projenin `.venv` Python'u kullanılır (bulunamazsa `py -3.11`). NumPy mevcut olmalıdır; proje ortamında zaten kullanılmıştır. Paket içindeki sabit pycocotools wheel'i yalnızca bu aracın `_vendor` klasörüne açılır. İnternet, pip kurulumu, ana `.venv` güncellemesi yok.

Kaynak veritabanları salt okunur açılır ve tam SHA256 değerleri başta/sonda denetlenir:
- `experiments/testdev_frozen_v1_seed42/predictions.sqlite3`
- `experiments/testdev1280_extension_v1_seed42/predictions.sqlite3`

Test-dev ham anotasyonlarının her birinin hash'i kontrol edilir. Eski22/23/24 raporlarının gerekli sabit kayıtları pakete dahildir; bunları arayıp yeniden kopyalamanız gerekmez.

## Kesinti ve saklama
Yeni kayıtlar `D:\HyCADO_Temiz_Baslangic\experiments\yolo800_1280_fusion_v1_seed42` altında tutulur. **Bu klasörü ve paketin açıldığı klasörü koruyun.** Elektrik kesilmesi/restart sonrasında aynı CMD'yi çalıştırın; tamamlanan tahmin, görüntü eşleştirmesi ve bootstrap çekilişleri doğrulanarak yeniden kullanılır. Yarım işlem SQLite tarafından geri alınır. Bir hata varsa klasörleri silmeyin; raporu gönderin.

Rapor ZIP'i özetleri, indeksleri, metrikleri ve bootstrap çekilişlerini içerir. Tam yeni tahmin/eşleştirme önbelleği deney klasöründeki `fusion_and_metrics.sqlite3` içindedir; özet ZIP bunun yedeği değildir.

## Kontrol ve süre
Başlangıçta otomatik sentetik testler çalışır. Üç aşama görünür: FUSION FREEZE, CPU MATCHING, PAIRED BOOTSTRAP. Her aşamanın kalan süre tahmini kendi ölçümünden üretilir; toplam süre tahmini değildir. Gerçek1610 görüntülük çalıştırma kullanıcının yerel arşivlerinde yapılacaktır; paket hazırlama sırasında gerçek deney sonucu üretilmemiştir.

Makaleyi veya GitHub'ı otomatik değiştirmez. Önce rapor birlikte değerlendirilir.
