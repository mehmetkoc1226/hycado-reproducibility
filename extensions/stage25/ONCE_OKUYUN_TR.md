# HyCADO 25 — Son konfigürasyonun nitel görselleri

**Model: 6.1 Sol · Extra High.** Bu adım mevcut dosyalardan görsel üretir.

**Düzeltme 1:** Windows'ta testlerin geçici SQLite dosyalarını silememesine neden olan açık bağlantılar düzeltildi. ZIP'i yeni ve boş bir klasöre çıkartın; önceki indirilen klasörle karıştırmayın. Ayrıntılar `DUZELTME_1_TR.md` dosyasındadır.

1. ZIP'i **tamamen ayrı bir klasöre çıkartın**. Proje klasörüne kopyalamayın; ZIP'in içinden çalıştırmayın.
2. `25_NITEL_GORSELLERI_HAZIRLA.cmd` dosyasına çift tıklayın.
3. İş bitince aşağıdaki dosyayı bu sohbete gönderin:

   `D:\HyCADO_Temiz_Baslangic\experiments\qualitative1280_v1_seed42\25_nitel_gorsel_rapor.zip`

Ana görselin önizlemesi aynı klasörde `qualitative1280_main_preview.png` olarak oluşur. ZIP gerçek görüntüleri, orijinal anotasyonları, donmuş tahminleri, 12 karşılaştırma önizlemesini ve 12 bağlam görüntüsünü içerir. Gönderimden sonra son şekli ve İngilizce açıklamasını makale düzenine göre hazırlayacağız.

**Eğitim, GPU çıkarımı, yeni metrik hesabı ve paket kurulumu yoktur.** Var olan Pillow ve NumPy kullanılır; ana ortam değiştirilmez. Eski veritabanları salt okunur açılır. Yalnızca yeni `qualitative1280_v1_seed42` klasörüne sonuç yazılır. Daha önceki 20 ve 24 klasörlerini koruyun.

Süre 12 görüntünün okunması ve PNG sıkıştırmasına bağlıdır; genellikle birkaç dakika mertebesinde olması beklenir, hedef bilgisayarda henüz ölçülmemiştir. Yeniden çalıştırılabilir; başarılı önceki ZIP, yenisi tamamlanana kadar korunur.

## Görsellerde ne var?

- Sütunlar: **GT / YOLO-1280 / HyCADO-1280 / SAHI-640**.
- Görüntüler 21. adımda belirtilen 20261112 tohumu ve SHA256 kuralının aşağıda somutlaştırılmasıyla seçilir; tahmin başarısı seçimde kullanılmaz.
- 12 farklı dosya adı temelli gruptan 12 görüntü seçilir. Ana görsel sıralamadaki ilk 3 görüntüdür; sonraki 9'u tamamlayıcı örneklerdir. Bunlar doğrulanmış bağımsız sahne grupları değildir.
- Tüm yöntemlerde aynı 640×420 doğal piksel penceresi ve **skor ≥0.05** gösterim eşiği kullanılır. Skorlar yeniden kalibre edilmez; bu ortak sayısal eşik aynı işletim noktası/yanlış pozitif oranı anlamına gelmez. AP veya Recall eğrisi yerine geçmez.
- Pencere, normal GT merkezlerinin en yoğun olduğu adaydan belirlenir; bu **GT kullanan bir sunum seçimidir**, çıkarım işlemi değildir. Pencere seçimi tahmin başarısından bağımsızdır.
- 22–24'teki sabit ignore filtresi GT ve tahminlere uygulanır. Ham anotasyonlar ve tam donmuş tahminler ayrıca aynen saklanır. 1000 aday gibi yeni bir küresel kesme uygulanmaz.
- Renkler sınıfı belirtir; TP/FP veya doğruluk etiketi değildir. Çerçeve sınırındaki kutular yalnızca resmedilirken kesilir. Eşleştirme sonucu üretilmez.

## Seçim kuralının tam tanımı

Her görüntü için UTF-8 olarak `20261112|grup_etiketi|images/test-dev/dosya.jpg` metninin SHA256 değeri hesaplanır. Değerler artan sıraya konur; her gruptan ilk karşılaşılan görüntü tutulur ve ilk 12 farklı grup alınır. Ana görsel ilk 3'tür. Kural, son görseller görülmeden dondurulmuştur; grup örneklem olasılıkları eşit değildir ve nicel genelleme iddiası yoktur.

Pencere adayları normal GT'nin anotasyon sırasından en fazla 100 eşit aralıklı indisle seçilir (`floor(i*(N−1)/(n−1))`). Her aday nesne merkezi çevresine 640×420 pencere konur ve görüntü sınırına ötelenir. İçerideki normal GT merkezi sayısı en fazla olan pencere alınır; eşitlikte küçük x, ardından küçük y seçilir. GT yoksa orta pencere kullanılır.

## Hata olursa

CMD'nin yanındaki `25_nitel_gorsel_hata.txt` dosyasını gönderin. Eski verileri silmeyin. Projenin yolu farklıysa:

```bat
25_NITEL_GORSELLERI_HAZIRLA.cmd "D:\HyCADO_Temiz_Baslangic"
```

Veri klasörü taşındıysa ikinci argüman, `images` ve `annotations` alt klasörlerinin bulunduğu test-dev dizinidir.

Bu adımın ardından bilimsel yorum ve makale güncellemesi için **Astra · Extra High** kullanacağız; geçiş zamanı ayrıca belirtilecek.
