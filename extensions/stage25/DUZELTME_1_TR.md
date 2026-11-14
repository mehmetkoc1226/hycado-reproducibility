# 25 — Windows dosya kilidi düzeltmesi

Önceki paketin testlerinde `with sqlite3.connect(...)` bağlantıyı kapatmıyordu. Bu yapı yalnızca işlemi commit/rollback eder. Linux açık dosyayı silebildiği için eski testler bunu yakalamadı; Windows geçici veritabanını silerken WinError 32 verdi.

- Tüm test bağlantıları işlem bitince açıkça kapatılır.
- Her test sonunda açılan bütün bağlantıların kapandığı ayrıca kontrol edilir. Bu kontrol, eski kodun hatasını Linux üzerinde de yakalar.
- Salt okunur bağlantının ilk ayarında hata çıkması ve görsel hazırlama yarıda kesilmesi durumunda da bağlantı kapanışı sınanır.
- Testler atlanmaz; hata veya geçici dosya temizleme sorunları bastırılmaz.

**Görüntü seçimi, tahminler, gösterim eşiği, pencere kuralı ve deney protokolü aynıdır.** Eğitim veya yeniden çıkarım yoktur. Önceki hata gerçek verilerin okunmasından önceki test aşamasındadır.

Güncel ZIP'i **yeni ve boş bir klasöre** çıkartıp `25_NITEL_GORSELLERI_HAZIRLA.cmd` dosyasını çalıştırın. Eski indirilen klasörle dosyaları karıştırmayın. Projedeki deney klasörlerinden hiçbirini silmeyin. Gönderilecek sonuç dosyasının yolu değişmedi:

`D:\HyCADO_Temiz_Baslangic\experiments\qualitative1280_v1_seed42\25_nitel_gorsel_rapor.zip`

Düzeltme yerel Linux ortamında bağlantı ömrü kontrolleriyle doğrulanmıştır; bu ortamda Windows çalıştırılmamıştır.
