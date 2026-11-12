# HyCADO · 18 Test-dev doğrulaması

Bu paket, önceki VisDrone VAL çalışmasındaki HyCADO v1'i **dondurulmuş ağırlık ve ayarlarla** VisDrone2019 DET test-dev'de doğrular. Bu bölüm daha önce ağırlık, eşik veya ayar seçimi için kullanıldıysa, tam koşuyu bağımsız doğrulama olarak başlatmayın.

## Çalıştırma

1. Bu paketi tamamen bir klasöre çıkarın. Klasörü mevcut projeye kopyalamanız gerekmez.
2. [VisDrone veri setinin resmî sayfasından](https://github.com/VisDrone/VisDrone-Dataset#task-1-object-detection-in-images), **Task 1 → testset-dev (GT available)** indirin. [Doğrudan Google Drive bağlantısı](https://drive.google.com/file/d/1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V/view). Başka bir dosya (özellikle testset-challenge) kullanmayın. `VisDrone2019-DET-test-dev.zip` dosyasını CMD dosyalarının yanına koyun. Windows Downloads veya proje içindeki `data\raw` klasöründeyse paket onu da bulur. ZIP'i açmanız gerekmez; isterseniz `images` ve `annotations` içeren açılmış klasörü ikinci argümanla verebilirsiniz.
3. **`18A_TESTDEV_ON_KONTROL.cmd`** dosyasını çalıştırın. Üç eski VAL görüntüsünde YOLO-800, YOLO-1280 ve HyCADO çıktıları dondurulmuş çıktılarla birebir eşleşmeli. Sonra 12 test-dev görüntüsünde, etiketleri açmadan, çıkarım süresini tahmin eder. **GPU tahmini, CPU puanlama ve güven aralığı süresini içermez.** `PREFLIGHT_OK` durumunda özet dosyası `18_testdev_onkontrol.json` olur.
4. Ön kontrol başarılıysa ve süre uygunsa **`18B_TESTDEV_TAM_DOGRULAMA.cmd`** dosyasını çalıştırın. Bu bölümü daha önce model/ayar seçmek için kullanmadıysanız `EVET` yazın. Kesintiden sonra aynı CMD'yi yeniden açın. Tamamlanan her görüntü-yöntem tahmini, etiket eşleştirmesi ve bootstrap tekrarı korunur.
5. Tamamlanınca **`D:\HyCADO_Temiz_Baslangic\experiments\testdev_frozen_v1_seed42\18_testdev_dogrulama_rapor.zip`** dosyasını paylaşın. `predictions.sqlite3` dosyasını yerelde koruyun; raporu paylaşmak için onu göndermeniz gerekmez. Bir hata olursa ekrandaki mesajı veya aynı klasördeki `preflight_error.json` / `full_error.json` dosyasını paylaşın.

Varsayılan proje klasörü `D:\HyCADO_Temiz_Baslangic`. Farklı bir konum için CMD'de aşağıdaki biçimi kullanın; ikinci argüman isteğe bağlıdır:

```bat
18A_TESTDEV_ON_KONTROL.cmd "D:\Projem" "D:\Veri\VisDrone2019-DET-test-dev.zip"
18B_TESTDEV_TAM_DOGRULAMA.cmd "D:\Projem" "D:\Veri\VisDrone2019-DET-test-dev.zip"
```

Projenin mevcut `.venv` ortamı ve önceki VAL dosyaları gereklidir. Paket kurmaz, ağırlık indirmez, MATLAB/Octave çağırmaz. Eski deneyleri veya verileri değiştirmez. Windows GPU denetimi için diğer GPU işlerini kapatın; bir koşu boyunca sürücüyü veya paket sürümlerini değiştirmeyin.

## Sabit deney

- YOLOv9-C 80. tur EMA, 800 ve 1280 giriş boyutunda. HyCADO, YOLO-800'ü ve aynı crop RT-DETR-L 80. tur EMA'yı kullanır. Crop çıkarımı 960, batch 1; `α=0.25`, sınıf bazlı NMS ve mevcut çıktı sınırları korunur.
- Üç yeniden yerleştirme tohumu: **0, 1, 2**. Görüntü başına crop sayısı ve sıralı genişlik/yükseklikler aynı; konumlar geçerli alan içinde eşit olasılıklı. Her üç seed'in AP'si ayrı hesaplanır. Birincil fark, HyCADO toolkit AP eksi bu üç AP'nin aritmetik ortalamasıdır.
- 1.610 görüntü/etiket çifti beklenir. TRAIN/VAL ile birebir dosya adı ve JPEG SHA256 kesişimi kontrol edilir. Bu kontrol benzer veya ardışık karelerin farklı bölümlerde bulunmadığını ispatlamaz.
- Tüm altı çıktı kümesi sabitlenene kadar test-dev anotasyonları puanlama için okunmaz. Puanlama önceki referansla uyumlu Python VisDrone ölçümünü kullanır. Çıktı bir COCO mAP veya yeni bir MATLAB/Octave referans doğrulaması değildir.
- Birincil fark için 1.000 eşleştirilmiş bootstrap tekrarı, dosya adı/aynı JPEG özetiyle oluşturulan yaklaşık kümeleri kullanır. Güven aralığı sabit modeller ve konum tohumları için örnek bileşimi belirsizliğini gösterir; farklı eğitim tohumlarını ölçmez. Nokta sonuçları bootstrap'ten önce `metrics.csv` dosyasına yazılır.
- Pilot süre ölçümü, toplam işi öngörmek içindir; makalede gecikme veya FPS karşılaştırması olarak kullanılmaz.
