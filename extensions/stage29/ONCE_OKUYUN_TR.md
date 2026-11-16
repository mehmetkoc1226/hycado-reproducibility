# HyCADO29 — ham crop denetimi

1. ZIP'i tamamen bir klasöre çıkartın.
2. **29_HAM_CROP_DENETIMI.cmd** dosyasını çalıştırın.
3. Bitince **aynı CMD'nin yanındaki 29_ham_crop_denetim_rapor.zip** dosyasını gönderin.

İndirilenler'de çıkartmanız yeterlidir; ana proje klasörüne dosya kopyalamayın. Varsayılan proje `D:\HyCADO_Temiz_Baslangic`. Farklı bir yol için terminalden CMD'ye ilk argüman olarak verilebilir:

```bat
29_HAM_CROP_DENETIMI.cmd "D:\HyCADO_Temiz_Baslangic"
```

## Amaç
20 numaralı uzantıdaki `crop_raw` tablosunun tüm1610 görüntüsünü doğrular. Her crop'un300×14 ham sorgusunu, dtype/sonluluk/skor aralığını, crop geometrisini ve arşiv hash'ini denetler. Kayıtlı α=0.25 ile HyCADO-800, HyCADO-1280 ve global1280/regions800 sonuçlarını birebir yeniden kurar. Böylece α duyarlılığının yeniden çıkarım yapmadan hesaplanıp hesaplanamayacağını belirler.

Bu araç α taraması veya AP hesabı yapmaz. Model ağırlığı yüklemez, görüntü/anotasyon okumaz, GPU veya pip kullanmaz. Windows'taki mevcut Python3.11/NumPy ortamını kullanır. Hazırlık testleri çalışır; ardından tam kaynak hash'leri ve görüntü başına yeniden üretim denetlenir.

## Geometri raporu
Final HyCADO-1280 için, NMS öncesi crop adayları / NMS sonrasında kalan crop adayları / görsel gösterimde kullanılan skor≥0.05 alt grubu ayrı raporlanır. Kutu alanının ait olduğu crop alanına oranı, crop sınırlarına temas, sınıf sayıları ve sabit kuralla seçilmiş örnek koordinatları kaydedilir. Bu alt grup yalnızca gösterim skorunu uygular; anotasyonlar okunmadığından ignore bölgeleri burada filtrelenmez ve nitel şeklin tam gösterim kümesiyle aynı olduğu iddia edilmez. Sorgu ve crop kimliği korunur; eşit skorlarda sıralama değiştirilmez.

Crop'u kaplayan bir kutu otomatik olarak yanlış pozitif veya artefakt değildir. GT eşleştirmesi yapılmadığından bu hüküm verilmez. Bu rapor geometriyi ve kayıtların analiz için yeterliliğini gösterir.

## Saklama ve devam
Eski18/20 kaynak veritabanları salt okunur açılır; tam hash'leri başta/sonda kontrol edilir. Bütün yeni kayıtlar `experiments\raw_crop_audit_v1_seed42` altında tutulur. **Bu yeni klasörü, eski18/20 klasörlerini ve bu paketin çıkartıldığı klasörü koruyun.** Restart/kesinti olursa aynı CMD'yi yeniden çalıştırın; biten görüntüler doğrulanarak yeniden kullanılır.

Özet rapor ZIP'i büyük ham crop veritabanlarının yedeği değildir. Ham sorguların asıl kaynağı `experiments\testdev1280_extension_v1_seed42\predictions.sqlite3` dosyasıdır; bunu silmeyin.

Gerçek1610 görüntülük denetim sizin yerel arşivlerinizde çalışacaktır. Paket hazırlanırken sentetik verilerle kaynak bağlama, yeniden üretim, NMS, geometri ve kesinti sonrası devam testleri yapıldı; gerçek arşivler burada açılmış değildir.

Makale veya GitHub otomatik değiştirilmez.
