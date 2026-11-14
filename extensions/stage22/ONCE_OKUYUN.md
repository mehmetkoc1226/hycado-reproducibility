# HyCADO 22 v2 — SAHI aday sayısı kontrolü düzeltmesi

## Şimdi çalıştırılacak tek dosya

1. ZIP'i yeni ve ayrı bir klasöre tamamen çıkarın. Projenin veya v1 paketinin üzerine kopyalamayın.
2. **22C_COCO_DUZELT_VE_TAMAMLA.cmd** dosyasını çalıştırın.
3. Başarılı bitince CMD'nin yanında oluşan **22B_coco_nokta_sonuclar_rapor.zip** dosyasını gönderin.

22C, güncellenmiş 22A kontrolünü ve ardından tam 22B hesabını sırayla kendisi çalıştırır. Arada başka bir dosya açmanız gerekmez. Ön kontrol başarısız olursa tam hesaba geçmez; ekranda belirtilen 22A raporunu gönderin. Tam hesap başarısız olursa 22B raporunu gönderin.

## Düzeltmenin nedeni

v1 doğrulayıcısı, YOLO'ya ait 1000 kutu sınırını bütün yöntemlere uygulamıştı. Dondurulmuş SAHI-640 VAL indeksinde iki görüntü bu sayıyı aşar; en yüksek sayı 1018'dir. P2'nin kaynak kodu SAHI'nin birleştirilmiş arşivine böyle bir global sınır uygulamaz. Bu nedenle v1 değerlendirme kodu geçerli bir tahmini reddetti.

v2, SAHI satırlarının sayısını dondurulmuş indeksteki sayı ve dosya SHA256'sı ile birebir doğrular. Kutuları kesmez, sıralarını veya skorlarını değiştirmez. YOLO ve HyCADO kaynakları için 1000 kutu kontrolü korunur. Geometri, sayısal değer, sınıf, görüntü kimliği ve dosya bütünlüğü kontrolleri korunur.

Ön kontrolde artık beş P2 yönteminin 548'er VAL tahmin dosyasının tamamı okunur ve doğrulanır. Metrik motorunun küçük kontrolü yine her ayrımdan 12 sabit görüntü kullanır; bu küçük örneğin AP'si makale sonucu değildir.

## Deney protokolü ve çıktıların korunması

- Bilimsel metrik protokolü v1 ile aynı dosyadır. COCO cap100 ve ek cap500, IoU/alan aralıkları, ignore işlemesi ve yöntem listesi değişmedi.
- Yeni eğitim, GPU çıkarımı, eşik araması veya yeniden NMS yoktur.
- Eski kaynak veritabanları salt okunur açılır; eski deneylere veya ağırlıklara yazılmaz.
- Yeni sonuç/cache klasörü: **D:\HyCADO_Temiz_Baslangic\experiments\supplementary_coco_v2_seed42**.
- v1'in değerlendirme cache'i olduğu gibi kalır. İki kod sürümünün cache'leri karıştırılmaz. v1 başlangıçta durduğu için v2 değerlendirmeye ilk görüntüden başlar; dedektörler yeniden çalıştırılmaz.

Tam hesap 548 VAL görüntüsünde altı ve 1610 test-dev görüntüsünde sekiz benzersiz yöntemi değerlendirir. AP, AP50/75, küçük/orta/büyük nesne AP'si ve sınıf AP'leri hesaplanır. Bunlar nokta tahminleridir; eşleştirilmiş bootstrap güven aralıkları ve recall–FP analizleri sonraki aşamadır.

## Ortam

Varsayılan proje D:\HyCADO_Temiz_Baslangic. Kayıtlı Windows x64 / Python 3.11 ve mevcut NumPy kullanılır. Paketteki pycocotools 2.0.11 wheel'i yalnızca paket içindeki _vendor/ klasörüne açılır. pip veya internetten kurulum yapılmaz; ana .venv paketleri değiştirilmez.

Farklı konumlar için CMD'yi proje klasörü ve isteğe bağlı açılmış test-dev veri klasörü argümanlarıyla çalıştırabilirsiniz. Veri klasörü images/ ve annotations/ içermelidir. 22A/22B ayrı başlatıcıları da pakettedir; şu an yalnızca 22C yeterlidir.

## Kesinti ve saklanacak dosya

Elektrik kesintisi veya restart olursa aynı **22C** dosyasını yeniden çalıştırın. Küçük kontrol tekrarlanır; v2 tam hesapta tamamlanıp doğrulanmış görüntü/yöntem kayıtları tekrar hesaplanmaz.

Yeni sonuç klasöründeki **evaluation_cache.sqlite3** dosyasını silmeyin; sonraki analizlerde kullanılacak. Büyük cache dosyasını göndermeyin. Gönderilecek ZIP CMD'nin yanındadır.

Metrik adı: COCO-style AP on VisDrone with stated ignore-region preprocessing. Eski toolkit AP korunur. Bu COCO uyarlaması, farklı VisDrone dönüşümleri veya leaderboard skorlarıyla otomatik eşdeğerlik iddia etmez.

