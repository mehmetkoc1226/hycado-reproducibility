# 20 · Test-dev üzerinde sabit HyCADO-1280 karşılaştırması

## Çalıştırma

1. Bu ZIP'i ayrı bir klasöre tamamen çıkarın. Mevcut proje klasörünün üzerine kopyalamayın. `engine`, `evidence` ve Python dosyaları bir arada kalmalı.
2. Diğer GPU işlerini kapatıp **`20A_TESTDEV1280_ON_KONTROL.cmd`** dosyasını çalıştırın. Üç geliştirme görüntüsünde YOLO-1280 ve tam HyCADO-1280 çıktıları, 19. koşunun tahminleriyle birebir karşılaştırılır. Sekiz test-dev görüntüsünde iki varyant hazırlanır; eski HyCADO-800 çıktısının da aynen yeniden üretildiği kontrol edilir. Etiket puanlaması yapılmaz. Kalan çıkarım süresi yaklaşık olarak raporlanır; CPU puanlama ve bootstrap süresi bu tahmine dahil değildir.
3. Ön kontrol `PREFLIGHT_OK` ile bitince **`20B_TESTDEV1280_TAM_KOSU.cmd`** dosyasını çalıştırın. Bu koşu 1.610 test-dev görüntüsünü tamamlar, AP ve 1.000 tekrarlı eşleştirilmiş güven aralıklarını hesaplar. Kesinti olursa aynı CMD'yi yeniden açın; kayıtlar korunur.
4. Bittiğinde şu dosyayı paylaşın:

`D:\HyCADO_Temiz_Baslangic\experiments\testdev1280_extension_v1_seed42\20_testdev1280_rapor.zip`

Ön kontrol raporu aynı klasörde `20_onkontrol.json`; hata raporu `error.json`. Ham crop sorguları ve tahminler `predictions.sqlite3` içinde saklanır. Bu dosyayı yerelde koruyun; sonuç raporunu paylaşmak için büyük SQLite dosyasını göndermeniz gerekmez.

## Gerekli mevcut dosyalar

Paket proje kökünü varsayılan olarak `D:\HyCADO_Temiz_Baslangic` kabul eder. Projenin mevcut `.venv` ortamı, doğrulanmış ağırlıklar ve şu deney klasörleri gereklidir:

- `experiments\testdev_frozen_v1_seed42` — 18. deneyin tahmin kasası, test-dev verisi ve rapor dosyaları.
- `experiments\full1280_development_v1_seed42` — 19. deneyin tahmin kasası ve rapor dosyaları.

Bu klasörleri taşımayın veya silmeyin. Paket eski SQLite dosyalarını **salt okunur** açar; eski deneyleri değiştirmez. Yeni veri indirme, paket kurulumu veya eğitim yoktur.

Farklı proje konumu için:

```bat
20A_TESTDEV1280_ON_KONTROL.cmd "D:\Projem"
20B_TESTDEV1280_TAM_KOSU.cmd "D:\Projem"
```

18. deneyin açılmış test-dev verisini başka yere taşıdıysanız, `images` ve `annotations` içeren klasörü ikinci argüman olarak verin. ZIP dosyası değil, açılmış klasör beklenir:

```bat
20A_TESTDEV1280_ON_KONTROL.cmd "D:\Projem" "D:\Veri\VisDrone2019-DET-test-dev"
20B_TESTDEV1280_TAM_KOSU.cmd "D:\Projem" "D:\Veri\VisDrone2019-DET-test-dev"
```

## Sabit karşılaştırmalar

- **Birincil:** YOLO-1280'in hem bölgeleri seçtiği hem de nihai füzyonda kullanıldığı tam HyCADO-1280 eksi YOLO-1280 AP.
- **İkincil:** YOLO-800'ün seçtiği bölgelerdeki crop tahminlerinin YOLO-1280 ile füzyonu eksi YOLO-1280 AP.
- **İkincil:** İki 1280 kaskat varyantının AP farkı.
- Eski HyCADO-800 sonuçları ücretsiz bir referans olarak raporlanır; bunların her görüntüde ve toplu metriklerde 18. koşuyla aynı olduğu kontrol edilir.

YOLO-1280 ve YOLO-800 global tahminleri 18. koşudan kullanılır. Crop ağının 80. tur EMA ağırlıkları, 960 giriş, batch=1, `α=0.25` ve NMS kuralları korunur; ayar taraması yapılmaz. 18. koşu yalnızca son füzyon çıktılarını sakladığı için bastırılmış ham crop adayları geri alınamaz. Bu nedenle crop sorguları yeniden hesaplanır ve artık yeni kasada korunur. Aynı görüntüde iki planın birebir aynı dikdörtgenleri varsa tek batch-one ağ çıktısı iki füzyonda da kullanılır; crop sırası ve aday sırası her varyantta korunur.

Tüm yeni tahminler sabitlenmeden yeni etiket eşleştirmesi açılmaz. Ancak test-dev daha önce 18. koşuda incelenmiştir. Bu çalışma **önceden incelenmiş test-dev üzerinde sabit ayarlı, keşifsel genişletmedir**; tamamen dokunulmamış bir test veya yeni bir veri kümesi değildir. Güven aralıkları sabit modeller için yaklaşık dosya adı/hash kümelerini yeniden örnekler; farklı eğitim tohumlarını kapsamaz. Birincil ve ikincil karşılaştırmalar birlikte raporlanır; ikincil aralıklar çoklu karşılaştırmaya göre düzeltilmez.

AP, önceki VisDrone toolkit kurallarıyla uyumlu Python ölçümüdür; COCO mAP değildir ve bu koşuda MATLAB/Octave doğrulaması tekrar çalıştırılmaz. Önceki YOLO-1280 ve HyCADO-800 metrikleri yeniden elde edilmelidir. Süreler ilerleme tahmini içindir; makalede gecikme/FPS kıyası olarak kullanılmaz.
