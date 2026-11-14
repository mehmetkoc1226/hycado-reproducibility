# HyCADO 23 — COCO güven aralıkları ve recall–FP analizi

## Çalıştırma

1. ZIP'i yeni ve ayrı bir klasöre tamamen çıkarın. Ana proje üzerine kopyalamayın.
2. **23_COCO_GUVEN_VE_RECALL.cmd** dosyasını çalıştırın.
3. Bitince CMD'nin yanında oluşan **23_coco_guven_recall_rapor.zip** dosyasını gönderin.

Varsayılan proje D:\HyCADO_Temiz_Baslangic. Farklı proje konumu varsa CMD'yi proje klasörünü ilk argüman olarak vererek çalıştırabilirsiniz.

## Yapılan hesap

- 1610 test-dev görüntüsü; YOLO-1280, HyCADO-1280 ve global YOLO-1280 + 800'den seçilmiş crop bölgeleriyle karma varyant.
- Aynı dondurulmuş 61 dosya adı/hash grubu, aynı seed 20261102 ve 1000 eşleştirilmiş bootstrap tekrarı.
- Ana COCO karşılaştırması HyCADO-1280 eksi YOLO-1280 AP. İki önceden tanımlı ikincil fark da raporlanır.
- COCO güven aralıkları area=all ve maxDets=100 içindir: AP, AP50, AP75 ve AR100. Cap500 ve küçük/orta/büyük nesne AP nokta değerleri 22B'de korunur; bu pakette onlar için ayrıca bootstrap yapılmaz.
- IoU 0.50/0.75, görüntü başına 0.5/1/2/5/10 FP bütçesinde mikro recall ve güven aralıkları. Skorları eşit kutular aynı anda alınır veya dışarıda bırakılır. Ana betimsel FP bütçesi 2'dir.
- Recall–FP, önceki global-max500 toolkit uyumlu eşleşmelerini kullanır. COCO'nun sınıf başına cap100 AR değeri ile aynı metrik değildir. Eğri eşikleri analiz içindir; yeni bir dağıtım eşiği seçilmez.

Bu işlem yeni eğitim, GPU çıkarımı veya hiperparametre araması değildir. Yeni model ağırlığı yüklenmez.

## Başlamadan otomatik doğrulama

14 sentetik/referans testi hedef bilgisayarda tekrar çalışır. Kod ve kaynak kayıtları doğrulanır. COCO cache'i yeniden birleştirilerek AP/AP50/AP75/AR değerlerinin tamamlanan 22B sonuçlarıyla eşleştiği kontrol edilir. Recall için dondurulmuş TP/FP sayıları eski sonuçlarla karşılaştırılır. Eşleşme başarısızsa uzun bootstrap döngüsü başlamaz.

Tam tekrar hesabında görüntü kopyaları, eşit skorlar içindeki kararlı sırayı korur; 101 recall noktalı COCO AP hesabı kullanılır. Recall eğrileri bütün skor eşitliklerini birlikte ele alır.

## Gerekli kaynak dosyalar

Bu dosyalar mevcut yerlerinde kalmalı:

- experiments/supplementary_coco_v2_seed42/point_results.json
- experiments/supplementary_coco_v2_seed42/evaluation_cache.sqlite3
- experiments/testdev1280_extension_v1_seed42/predictions.sqlite3

Görüntüleri veya anotasyonları bu pakete kopyalamayın. Kaynak veritabanları salt okunur açılır. 20/22 değerlendirmelerini aynı anda yeniden çalıştırmayın; bu kaynaklara yazan bir süreç olmamalı.

Windows x64 / Python 3.11 ve 22B ile aynı NumPy ortamı kullanılır. Paketteki pycocotools wheel'i yalnızca paket içindeki _vendor/ klasörüne açılır. İnternetten veya ana .venv ortamına paket kurulmaz.

## Süre ve kesintiden devam

Bu, önceki 70 dakikalık nokta hesabından farklı bir iştir; onun süre tahmini buraya taşınmaz. İlk 5 bootstrap tekrarından sonra cihazınıza göre yaklaşık kalan süre yazılır. Tekrar sayısı deney protokolü gereği 1000'dir.

Tamamlanan her tekrar yeni bootstrap.sqlite3 dosyasına kaydedilir. Restart veya kesinti sonrası aynı CMD'yi yeniden çalıştırın. Kaynak doğrulaması/derleme tekrarlanır, tamamlanan bootstrap tekrarları korunur. Sonuçları veya dosyaları elle düzenlemeyin.

Yeni çıktı klasörü: **D:\HyCADO_Temiz_Baslangic\experiments\coco_uncertainty_recall_v1_seed42**.

Bu klasördeki **bootstrap.sqlite3** dosyasını ve önceki **evaluation_cache.sqlite3** dosyasını koruyun. Büyük veritabanlarını göndermeyin; gönderilecek ZIP CMD'nin yanındadır.

## Çıktılar ve makale grafikleri

Rapor intervals.json, recall_curves.csv, recall_budgets.csv, bootstrap_draws.csv ve kaynak/protokol kayıtlarını içerir. Bu sayılarla sonraki değerlendirmede güven aralığı ve recall–FP grafikleri üretilecek. Hız/FPS veya Pareto analizi yapılmaz.

Güven aralıkları sabit eğitimli modeller altında örnek bileşimi belirsizliğini ölçer. Eğitim seed'i değişkenliğini ölçmez. Dosya adı/hash grupları doğrulanmış bağımsız sahneler değildir; test-dev daha önce incelendiği için analiz keşifseldir. İkincil aralıklar çoklu karşılaştırma düzeltmesi yapılmamış betimsel aralıklardır.
