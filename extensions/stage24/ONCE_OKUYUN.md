# HyCADO 24 — SAHI-640 test-dev karşılaştırması

## Şimdi yapılacak

1. ZIP'i yeni ve ayrı bir klasöre tamamen çıkarın. Ana projenin üzerine kopyalamayın.
2. Diğer GPU eğitim/çıkarım işlerini kapatın.
3. **24_SAHI640_TESTDEV.cmd** dosyasını çalıştırın.
4. Bitince CMD'nin yanında oluşan **24_sahi640_testdev_rapor.zip** dosyasını gönderin.

Bu yeni eğitim değildir. Aynı dondurulmuş YOLO ağırlığıyla gerçek SAHI kütüphanesi kullanılarak yalnızca SAHI-640'ın yeni test-dev tahminleri üretilecek. YOLO-1280 ve HyCADO-1280 yeniden çalıştırılmayacak; doğrulanmış eski sonuçları ve bootstrap tekrarları kullanılacak.

## Otomatik sıra

1. Paket, mevcut 22B/23 kayıtları, model ağırlığı, ortam ve bütün test-dev görüntü kimlikleri doğrulanır. 13 test hedef bilgisayarda tekrar çalışır.
2. Önceki VAL arşivinden beş görüntüde gerçek GPU SAHI çıktısı yeniden üretilir. Kutu sıralaması için aynı eski P2 canonical sıralama uygulanır; kutu/skorların sayısal değerleri arşivle birebir eşleşmelidir. Eski sıralama tanısındaki görüntü ve 1018 kutulu yoğun görüntü bu kontrole dahildir. Burada GT açılmaz veya AP hesaplanmaz.
3. Kontrol geçerse 1610 test-dev görüntüsünün tamamında çıkarım yapılır. Her tamamlanan görüntü hemen yeni veritabanına kaydedilir.
4. Bütün tahminler kaydedilip prediction_index.json dondurulduktan sonra yeni etiket satırları açılır; CPU ile toolkit AP ve COCO metrikleri hesaplanır.
5. Aynı 61 grup ve seed ile 1000 SAHI COCO bootstrap tekrarı hesaplanır. Karşılaştırma için önceki 23'ün YOLO/HyCADO tekrarları aynı draw kimliğiyle eşleştirilerek kullanılır.

Kaynak eşliği başarısız olursa program tam çıkarıma geçmez. Hata raporunu gönderin; dosyaları, eşikleri veya kontrol koşullarını elle değiştirmeyin.

## Sabit ayarlar

- Ağırlık: release_stage/checkpoints/yolo_epoch_0080.pt, aynı seed42 son EMA kaydı; SHA256 protokolde.
- Gerçek SAHI 0.11.35, aynı Ultralytics 8.3.203 ve PyTorch 2.6.0+cu124.
- Crop dilimleri 640×640; örtüşme her eksende 0.2; model girişi 800; conf 0.001; tam görüntü standart geçişi açık.
- P2'de kullanılan SAHI model oluşturma/çıkarım fonksiyon gövdeleri değiştirilmeden korunmuştur. Yeni FP16, TensorRT, batch veya postprocess ayarı eklenmez.
- NMS, IOU eşleştirme, eşik 0.5 ve sınıfa duyarlı birleştirme; aynı kütüphane varsayılanları korunur.
- Birleştirilmiş SAHI tahmin arşivi 1000 kutuda kesilmez. Toolkit metrik tanımında global500; COCO'da sınıf başına cap100 ve ayrıca cap500 ayrı uygulanır.
- Yeni eğitim, eşik seçimi, ayar taraması veya hız/FPS kıyaslaması yoktur. Ekrandaki süreler yalnızca ilerleme tahminidir.

## Sonuçlar

metrics.json ve metrics.csv hem eski toolkit AP'yi hem COCO AP/AP50/AP75/küçük-orta-büyük AP değerlerini içerir. Yeni çıkarım yalnızca SAHI içindir; karşılaştırıcı değerler kaynak kimliği doğrulanmış 22B ve 20 kayıtlarından gelir.

Yeni ana güven aralığı: **HyCADO-1280 eksi SAHI-640, COCO AP cap100, area=all**. SAHI-640 eksi YOLO-1280 ikincil karşılaştırmadır. AP50/AP75/AR100 aralıkları da kaydedilir. Bu pakette toolkit AP için ayrıca bootstrap yapılmaz; bu nokta tablosu korunur. Cap500 ve nesne boyutu AP'leri nokta sonuçlarıdır.

Güven aralıkları sabit modeller altında örnek bileşimi belirsizliğini ölçer. 61 dosya adı/hash grubu doğrulanmış bağımsız sahneler değildir; daha önce incelenmiş test-dev üzerinde analiz keşifseldir. İkincil aralıklar çoklu karşılaştırma düzeltmesi yapılmamış betimsel aralıklardır.

## Dosyalar ve ortam

Varsayılan proje: D:\HyCADO_Temiz_Baslangic. Farklı proje yolu ilk CMD argümanıyla verilebilir. Açılmış test-dev klasörü ikinci argümanla verilebilir; normalde 18. adımın mevcut dataset_... klasörü otomatik bulunur.

Kritik mevcut kaynaklar:

- release_stage/checkpoints/yolo_epoch_0080.pt
- experiments/p2_strong_baselines_v1_seed42/sahi_vendor
- Aynı klasörde official_val_protocol.json, official_val_prediction_index.json ve official_val_predictions/sahi640 dosyaları
- data/prepared/images/val içindeki eski kontrol görüntüleri
- 18. adımın açılmış test-dev görüntü/anotasyon klasörü
- experiments/testdev1280_extension_v1_seed42/predictions.sqlite3
- experiments/supplementary_coco_v2_seed42/point_results.json
- experiments/coco_uncertainty_recall_v1_seed42/result.json ve point.json

Ana .venv değiştirilmez; pip çalıştırılmaz. Mevcut P2 sahi_vendor kullanılır. Paket içindeki aynı pycocotools wheel'i sadece paket içindeki _vendor/ klasörüne açılır.

Yeni çıktılar **experiments/sahi640_testdev_v1_seed42** altında saklanır. Eski deney veritabanları ve ağırlıklar değiştirilmez. 20/22/23 kaynaklarına yazan başka bir koşuyu eşzamanlı başlatmayın.

## Süre ve kesinti

İlk birkaç test-dev görüntüsünden sonra çıkarım için yaklaşık kalan süre yazılır. Ardından ayrı CPU metrik ve bootstrap aşamaları başlar. Bu işlemin süresini önceki 22/23 süreleriyle eşitlemeyin.

Restart veya kesinti olursa aynı CMD'yi tekrar çalıştırın. Kaynak/VAL kontrolleri tekrarlanır; tamamlanan test-dev tahminleri, CPU eşleşmeleri ve bootstrap tekrarları korunur. Yeni **predictions.sqlite3** dosyasını silmeyin. Büyük veritabanını göndermeyin; gönderilecek rapor ZIP'i CMD'nin yanındadır.
