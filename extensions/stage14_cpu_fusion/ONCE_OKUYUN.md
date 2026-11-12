# YOLO-1280 ve mevcut crop tahminleri

ZIP'i açın; **14_YOLO1280_CPU_FUZYON.cmd** dosyasını çalıştırın. Dosyalar
aynı klasörde kalsın. Projeye kopyalamak veya paket kurmak gerekmiyor.

Bu CPU kontrolü mevcut 647 TRAIN-içi geliştirme görüntüsünün tahminlerini okur.
YOLO-1280 kutularını, YOLO-800'ün seçtiği mevcut 10.682 bölgeye ait kayıtlı
RT-DETR-L kutularıyla birleştirir. Crop skor çarpanı 0.25, sınıf bazlı NMS
IoU eşiği 0.70 ve değerlendirme sınırı 500 sabittir. Yeni eşik veya bütçe
aranmaz. Bu, bölgeleri YOLO-1280'in seçtiği tam HyCADO-1280 koşusu değildir.

Dört sonuç hesaplanır: YOLO-1280, crop eklemeden aynı NMS uygulanmış YOLO-1280,
eski HyCADO-800 (tüm crop'lar, 0.25), ve YOLO-1280 + eski crop'lar. İlk ve üçüncü
sonucun önceki raporlarla eşleşmesi zorunludur. Metrik mevcut VisDrone
toolkit uyumlu AP'dir. Resmi VAL okunmaz; gecikme ölçülmez.

GPU, yeni çıkarım ve eğitim yok. Eski deneyler korunur. Tamamlanan görüntülerin
CPU eşleşme kayıtları doğrulanarak yeniden kullanılır; kesilirse aynı CMD'yi
çalıştırın. 25 görüntüde bir ilerleme yazılır. Süre CPU ve disk hızına bağlıdır.

Gönderilecek rapor:
D:\HyCADO_Temiz_Baslangic\experiments\p4_yolo1280_cpu_fusion_v1_seed42\14_yolo1280_cpu_fuzyon_rapor.zip

Kaynak dosya eksik veya değişmişse program ilgili dosyayı belirterek durur ve
hata raporu üretir. Önceki deneyleri yeniden başlatmayın; önce raporu gönderin.
