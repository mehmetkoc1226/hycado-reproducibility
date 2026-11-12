# HyCADO-1280 geliştirme deneyi

## Şimdi yapılacak işlem

1. ZIP'i bir klasöre tamamen çıkarın. **Mevcut projenin üzerine kopyalamayın.** İçindeki `engine` ve `evidence` klasörleri aynı yerde kalmalı.
2. **`19A_TAM1280_ON_KONTROL.cmd`** dosyasını çalıştırın. Üç YOLO-1280 görüntüsünde eski tahminlerle birebir eşleşme aranır; sekiz görüntüde tam HyCADO-1280 çıktısı, önceki hız pilotunun çıktı özetleriyle karşılaştırılır. Kalan çıkarım süresi yaklaşık olarak raporlanır.
3. Ön kontrol başarılıysa **`19B_TAM1280_GELISTIRME.cmd`** dosyasını çalıştırın. 647 geliştirme görüntüsündeki tahminler tamamlanır, sonra AP ve güven aralığı hesaplanır. Kesinti olursa aynı CMD'yi yeniden çalıştırın. Tamamlanan tahminler ve CPU hesapları korunur.
4. Bittiğinde şu raporu paylaşın:

`D:\HyCADO_Temiz_Baslangic\experiments\full1280_development_v1_seed42\19_tam1280_gelistirme_rapor.zip`

Ön kontrol raporu aynı klasördeki `19_onkontrol.json`; hata raporu `error.json`. Tahminlerin bulunduğu `predictions.sqlite3` dosyasını da yerelde koruyun; rapor göndermek için bu büyük dosyayı paylaşmanız gerekmez.

Varsayılan proje yolu `D:\HyCADO_Temiz_Baslangic`. Farklıysa komut satırında:

```bat
19A_TAM1280_ON_KONTROL.cmd "D:\Projem"
19B_TAM1280_GELISTIRME.cmd "D:\Projem"
```

Yeni veri veya paket indirme, eğitim ve MATLAB/Octave gerekmiyor. Mevcut projenin `.venv` ortamı, ağırlıkları, P2 geliştirme tahminleri ve TRAIN ham anotasyonları kullanılır. Eski deneyler salt okunur.

## Ölçülen soru

Aynı 647 TRAIN içi geliştirme görüntüsünde:

- **YOLO-1280:** P2'nin dondurulmuş tahminleri aynen kullanılır. AP'nin 31.6513051607 olduğu yeniden doğrulanır.
- **YOLO-1280 + yalnızca son NMS:** Crop eklemeden yeniden NMS'nin etkisini kontrol eder.
- **Tam HyCADO-1280:** Hem crop bölgelerini seçmek hem de nihai füzyon için YOLO-1280 çıktısı kullanılır. Mevcut crop RT-DETR-L EMA ağırlıkları, 960 crop girişi, batch=1, `α=0.25` ve NMS ayarları değiştirilmez.

Eski +0.819 AP sonucu, YOLO-800'ün seçtiği crop bölgelerindeki tahminleri YOLO-1280 ile birleştiriyordu. Bu deney bölgeleri de YOLO-1280'den üretir. Eski 32 görüntülük hız pilotu tam tahminleri saklamadığı için crop çıkarımı yeniden yapılır; 32 görüntünün saklanan çıktı özetleri kontrol amacıyla kullanılır. YOLO-1280'in 647 görüntülük tahminleri yeniden hesaplanmaz.

Önce tüm tahminler sabitlenir, sonra ham anotasyonlar okunur. Ana ölçüm önceki toolkit ile uyumlu Python AP'dir; COCO mAP değildir. Birincil fark tam HyCADO-1280 eksi YOLO-1280 AP'dir. 1.000 eşleştirilmiş, dosya adı/aynı görüntü özetine dayalı yaklaşık küme bootstrap tekrarı kullanılır. Farklı eğitim tohumları ölçülmez.

Bu geliştirme bölümü daha önce incelenmiştir; sonuç bağımsız doğrulama değildir. Başarılı sonuç otomatik olarak VAL/test-dev koşusu başlatmaz. Bu koşuda VAL ve test-dev görüntüleri veya etiketleri açılmaz. İlerleme süreleri kontrollü gecikme/FPS sonucu olarak kullanılmaz.
