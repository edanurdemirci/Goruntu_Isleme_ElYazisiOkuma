# El Yazısı Tanıma ve Dijitalleştirme Projesi (Handwritten Text Recognition)

Bu proje, el yazısı içeren görüntüleri işleyerek içerisindeki metni dijital karakterlere dönüştüren **Yapay Zeka (CNN)** ve **Görüntü İşleme** tabanlı bir web uygulamasıdır.

Python, TensorFlow (Keras) ve Flask kullanılarak geliştirilmiştir.

![Proje Ekran Görüntüsü](screenshot.png)
<img width="1063" height="581" alt="image" src="https://github.com/user-attachments/assets/546444f6-c74e-412a-bbf0-66e771336608" />


##  Özellikler

* **Görüntü İşleme:** OpenCV kullanılarak yüklenen resimdeki gürültüler temizlenir ve her bir harf/karakter ayrı ayrı tespit edilir (Contour Detection).
* **Yapay Zeka Modeli:** Kendi eğittiğimiz CNN (Convolutional Neural Network) modeli ile 35 farklı sınıf (0-9 Rakamlar ve A-Z Harfler) %90+ doğrulukla tanınır.
* **Web Arayüzü:** Flask ile oluşturulan kullanıcı dostu arayüz sayesinde kolayca resim yüklenebilir ve sonuç anlık olarak görülür.
* **Anlık Görselleştirme:** Modelin tespit ettiği harfler, orijinal resim üzerinde kutucuklar içine alınarak kullanıcıya gösterilir.

##  Kullanılan Teknolojiler

* **Dil:** Python 3.10+
* **Web Framework:** Flask
* **Derin Öğrenme:** TensorFlow, Keras
* **Görüntü İşleme:** OpenCV, Imutils
* **Veri İşleme:** NumPy, Pandas

##  Kurulum ve Çalıştırma

Bu projeyi kendi bilgisayarınızda çalıştırmak için gerekli kütüphaneleri indirip aşağıdaki linke tıklayabilirsiniz.
Denemeniz için örnek bir el yazısı aşağıya eklenmiştir.



<img width="691" height="465" alt="image" src="https://github.com/user-attachments/assets/e111f439-5e26-4bb4-bfe7-db3e0eec0ddc" />


### Projeyi Klonlayın
```bash
git clone [https://github.com/edanurdemirci/Goruntu_Isleme_ElYazisiOkuma] 


