# HyCADO30 — α duyarlılığı ve crop kutularının GT eşleşmesi

1. ZIP'i tamamen çıkartın.
2. **30_ALPHA_DUYARLILIK.cmd** dosyasını çalıştırın.
3. Bitince **aynı CMD'nin yanındaki 30_alpha_duyarlilik_rapor.zip** dosyasını gönderin.

İndirilenler'de çıkartmanız yeterlidir. Dosyaları ana projeye kopyalamanız gerekmez. Proje varsayılanı `D:\HyCADO_Temiz_Baslangic`. Değişik proje/veri kökü için:

```bat
30_ALPHA_DUYARLILIK.cmd "D:\HyCADO_Temiz_Baslangic" "D:\Veri\VisDrone2019-DET-test-dev"
```

İkinci argüman, `annotations` alt klasörünü içeren test-dev köküdür. Veriler taşınmadıysa argüman gerekmez.

## Ne hesaplanacak?
Ham29 denetimi tamamlandıktan sonra sabitlenen beş değer: **α=0,0.125,0.25,0.5,1**. α=0, crop dalı kapalı orijinal YOLO-1280 sonucudur; skoru sıfırlanmış crop kutuları eklenmez. Diğer dört değer için ham crop sorguları aynı şekilde çözülür, yalnızca crop skoru çarpanı değişir; tüm adaylar yeniden sıralanır ve aynı NMS/1000 sınırı yeniden uygulanır. Eski final kutularının skorlarını değiştirmekle yetinilmez.

Önce bütün karşı-olgusal tahminler sabitlenir; sonra ham anotasyonlar okunarak önceki ignore kuralları ve pycocotools2.0.11 ile COCO AP, AP50/75, AP küçük/orta/büyük, AR ve10 sınıf AP hesaplanır. MaxDets100 ana,500 duyarlılık sonucudur. α=0.25, mevcut HyCADO-1280 çıktısını ve tamamlanmış22 metriklerini yeniden üretmek zorundadır. α=0'ın noktaları tamamlanmış22'den kullanılır; görüntü/çıktı bağları aynı sabit kaynakla doğrulanır.

Aynı çalıştırmada mevcut α=0.25 için crop alanının en az%95'ini kaplayan kutuların GT eşleşmesi ölçülür. Doğru/yanlış pozitif sayıları IoU0.5 ve0.75'te, ignore filtresinden sonra sınıf başına ilk100 tahmin üzerinden raporlanır. Gösterim skoru≥0.05 alt grubu da ayrıdır. Bu nedenle29'un bütün ham tahminleri içeren sayılarıyla aynı olmak zorunda değildir. Crop/sorgu kimliği, COCO'nun gerçek dtIds eşleşmesiyle korunur.

Bu, daha önce incelenmiş test-dev üzerinde **betimsel duyarlılık analizidir**. En yüksek sonucu bulup bağımsız test sonucu gibi sunmak için kullanılmaz. Orijinal α=0.25 sonucu korunur; makale veya final konfigürasyon otomatik değiştirilmez. Beş değerin tamamı raporlanır. Yeni bootstrap/güven aralığı hesabı yoktur; yalnızca nokta sonuçları ve tanısal eşleşme sayıları hesaplanır.

## Ortam
Mevcut Windows64-bit Python3.11 ve NumPy kullanılır. Sabit pycocotools wheel'i yalnızca bu aracın `_vendor` dizinine açılır. **Yeni eğitim, GPU çıkarımı, internet/pip kurulumu veya ana `.venv` güncellemesi yoktur.** Mevcut29 sonuç kaydı doğrulanır; gereken tamamlanmış rapor kayıtları pakette bulunmaktadır, ayrıca kopyalamanız gerekmez.

## Saklama ve kesinti
Kaynak: `experiments\testdev1280_extension_v1_seed42\predictions.sqlite3`; salt okunur açılır ve tam hash'i başta/sonda denetlenir. Bunu ve önceki deney klasörlerini koruyun.

Yeni çıktı: `experiments\alpha_sensitivity_v1_seed42`. **Bu klasörü ve paketin çıkartıldığı klasörü silmeyin.** Restart/kesinti sonrası aynı CMD'yi çalıştırın; tamamlanan görüntü/tahmin/eşleştirmeler doğrulanarak yeniden kullanılır. Bu çalıştırmanın tam yeni tahminleri `alpha_predictions_and_metrics.sqlite3` dosyasındadır; gönderilecek özet ZIP bunun yedeği değildir.

İlerleme iki faz gösterir: FROZEN ALPHA OUTPUTS ve CPU COCO MATCHING. Tahmin edilen kalan süre yalnızca o fazın ölçümüdür. Gerçek1610 görüntülük hesap burada yapılmadı; yerel arşivlerinizle çalışacaktır. Hazırlıkta sentetik örneklerle yeniden üretim, COCO-kutu kimliği, cap/ignore kuralları ve kaldığı yerden devam akışı sınandı.

Sonraki adım:30 raporunu28 ve29 ile birlikte değerlendirip makaleye duyarlılık sonuçlarını ve yeni füzyon kontrolünü işlemek. GitHub otomatik değiştirilmez.
