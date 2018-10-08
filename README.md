# AWS Kullanarak makina öğrenmesi modellerinin oluşturulması ve web servis olarak sunulması

Bu atölye çalışmasında AWS servislerini kullanarak adım adım bir makina öğrenmesi modeli oluşturacağız web servis olarak modelimizi web üzerinden erişime açacağız

### Ön koşullar

AWS hesabınız olmalıdır. Yoksa şu linkten oluşturabilirsiniz: https://aws.amazon.com/resources/create-account/

### Adımlar

Atölye çalışması esnasında aşağıdaki adımları tamamlayacağız:


*Amazon Sagemaker kullanarak notebook makinasının ayağa kaldırılması
*Amazon S3 üzerinde bucket'ımızın oluşturulması
*Modelin oluşturulması
*Modeli çağıracak AWS Lambda fonksiyonunun yazılması
*AWS Lambda fonksiyonunu web'e açmak için Amazon API Gateway servisinde konfigürasyonların yapılması
Servisin testi

## Amazon Sagemaker kullanarak notebook makinasının ayağa kaldırılması
*AWS Konsol'a login olun
*AWS Services etiketinin aldındaki arama kutusuna Sagemaker yazın, çıkan linke tıklayın
*Konsolda sağ üst köşedeki Region'lardan kendinize yakın olan birini seçin (sadece Amazon Sagemaker’ın kullanılabildiği Regionlar listelenmektedir). Seçtiğiniz Region’ın adını bir kenara not alın.

<p align="center">
<img src="Picture0.png">
<img src="Picture0.png">
<img src="https://github.com/barisyasin/sagemaker-intro-tr/blob/master/blob/master/Picture0.png">
<img src="https://github.com/awslabs/amazon-sagemaker-examples/blob/master/scientific_details_of_algorithms/lda_topic_modeling/img/img_documents.png">
</p>

* “Create notebook” instance butonuna tıkla

IMAGE 2

* Notebook instance name kutucuğuna istediğiniz tanımlayıcı metni girin. Notebook instance type olarak ml.m4.xlarge seçin. Diğer alanları değiştirmeden Create Notebook Instance butonuna tıklayın.

IMAGE 3

## Amazon S3 üzerinde bucket'ımızın oluşturulması
* Üzerinde makina öğrenmesi algoritmalarını deneyeceğimiz ortamımız oluşturulurken, bu ortamın ihtiyaç duyacağı veriyi tutacağımız S3 bucket’ımızı oluşturalım. Konsolda sol üst köşedeki Services butonuna tıklayın, çıkan kutucuğa S3 yazın ve en üstte gelen linke tıklayın.
* Gelen ekranda Create Bucket butonuna tıklayın

IMAGE 4

* Bucket name alanına içinde sagemaker kelimesi geçecek şekilde bir metin girin. Örn. adinizsoyadiniz-sagemaker. Bucket adını not alın.
* Region olarak yukarıda not aldığınız Region’ı seçin
* Sol alt köşede bulunan Create butonuna tıklayarak bucket’ınızı yaratın

## Modelin oluşturulması
*Konsolda sol üst köşedeki Services butonuna tıklayın, çıkan kutucuğa Sagemaker yazın ve en üstte gelen linke tıklayın
*Soldaki menüden Notebook instances linkine tıklayın
*Gelen listede oluşturduğunuz Notebook Instance’ın Status’u InService olarak görünüyorsa Open linkine tıklayın. InService olarak görünmüyorsa birkaç dakika daha bekleyin

IMAGE 5

*Open linkine tıkladığınızda tarayıcınızın yeni bir ekranında Jupyter açılacak
*Jupyter’de yukarıdaki tablardan SageMaker Examples’ı seçin
*Introduction to Applying Machine Learning bölmesindeki Breast Cancer Prediction.ipynb notebook’unu bulun ve en sağdaki Use butonuna tıklayın, gelen pop-up ekranında Create copy butonuna tıklayın. Tarayıcınız yeni bir ekranında seçtiğiniz Jupyter notebook açılacaktır

IMAGE 6

*Açılan notebook’u inceledikten sonra en üst kutucuktaki <your_s3_bucket_name_here> yazan yere not aldığınız bucket adını kopyalayın.

IMAGE 7

*En alttaki kutucuğa gidin. Satırın en başına # yazarak comment out edin.
*Cell menüsünde Run All’ı seçin. Notebook’daki bütün satırların çalışması bitene kadar bekleyin. Bu işlem yaklaşık 10 dakika sürecektir. O anda çalıştırılan kod satırlarının yanında (*) işareti çıkacaktır.

IMAGE 8

## Modeli çağıracak AWS Lambda fonksiyonunun yazılması

*Şimdi yarattığımız bu model’ı çağıran bir Lambda fonksiyonu yaratacağız. Amacımız aşağıdaki yapıyı kurarak modelimizi web servisler üzerinden ulaşılabilir hale getirmek

IMAGE 9

*Konsolda sol üst köşedeki Services butonuna tıklayın, çıkan kutucuğa Lambda yazın ve en üstte gelen linke tıklayın
*Gelen ekranda Create Function butonuna tıklayın
*Name alanına sagemakerinvokerfunction yazın
*Runtime’ı Python 3.6 seçin

IMAGE 10

*Role için Create a custom role seçeneğine tıklayın. AWS Lambda’nın Amazon Sagemaker’ı çağırırken kullanacağı rolü oluşturacağınız ekran açılacaktır. 
*Açılan ekranda View Policy Document’e tıkladıktan sonra ortaya çıkan Edit linkine tıklayın. Policy document kutucuğu editable duruma gelecektir. Kutucuğa şu text’i girin:
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": "sagemaker:InvokeEndpoint",
            "Resource": "*"
        }
    ]
}
*Allow butonuna tıklayın
*Create function butonuna tıklayın
*Yeni ekranda Function Code alanına doğru inin ve kodu aşağıdakiyle değiştirin

import os
import io
import boto3
import json
import csv

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    data = json.loads(json.dumps(event))
    payload = data['data']
    print(payload)
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=payload)
    print(response)
    result = json.loads(response['Body'].read().decode())
    print(result)
    pred = float(result['predictions'][0]['score'])
    print("pred: " + repr(pred))
    return pred

*Biraz daha aşağıdaki Environment Variables alanına gelin. Key alanına ENDPOINT_NAME yazın. Value alanına Notebook’unuzun çalışırken çıktı olarak ürettiği “DEMO-linear-endpoint-XXX” ile başlayan enpoint adını kopyalayın. Endpoint adını öğrenmek için tarayıcı ekranında açılan Notebook’unuza geri dönün. DEMO-linear-endpoint- metnini aratın. Bu metni bulduğunuz ilk kod kutucuğunun altında gördüğünüz üretilmiş endpoint’in adını kopyalayın (DEMO-linear-endpoint-201810060646 gibi bir metin olması gerekiyor). Kod kutucuğu hala işletilmemişse işletilmesini bekledikten sonra bu adımı gerçekleştirin.
*Save butonuna tıklayın

## AWS Lambda fonksiyonunu web'e açmak için Amazon API Gateway servisinde konfigürasyonların yapılması

*Şimdi oluşturduğumuz Lambda fonksiyonunu web servis olarak çağırabilmek için API Gateway servisini konfigüre edeceğiz.

*Konsolda sol üst köşedeki Services butonuna tıklayın, çıkan kutucuğa API Gateway yazın ve en üstte gelen linke tıklayın
*Create API butonuna tıklayın. API Name alanına sagemakerprediction gibi bir apı adı yazın, Endpoint Type’ını Regional olarak bırakın ve Create Api butonuna tıklayın.

IMAGE 11

*Actions menüsünden Create Resource’u seçin

IMAGE 12

*Resource Name alanına predictor yazın. Create Resource butonuna tıklayın

IMAGE 13

*Yine Actions menüsünden Create Method’u seçin

IMAGE 14

Açılan combo’dan POST metodunu seçin ve hemen yanındaki tick işaretine tıklayın

IMAGE 15

Lambda Function alanına bir önceki adımda yarattığınız lambda fonksiyonunun adını girin ve Save butonuna tıklayın. Çıkan ekranda OK’e tıklayın

IMAGE 16

Yine Actions menüsünden Deploy API’i seçin

IMAGE 17

Deployment Stage olarak  New Stage’i seçin ve ekran görüntüsündeki gibi doldurarak Deploy butonuna tıklayın

IMAGE 18

*Stages’in altındaki POST’a tıklayın. Karşınıza çıkan Invoke URL’I not alın

IMAGE 19

## Servisin testi
Postman gibi bir HTTP Client kullanarak oluşturduğunuz servisi test edin. Metod olarak post’u seçin. Request URL alanına yukarıda not ettiğiniz URL’i yapıştırın.  Alttaki Body tabından raw radio butonunu seçin

IMAGE 20

Örnek test datası:

{"data":"13.49,22.3,86.91,561.0,0.08752,0.07697999999999999,0.047510000000000004,0.033839999999999995,0.1809,0.057179999999999995,0.2338,1.3530000000000002,1.735,20.2,0.004455,0.013819999999999999,0.02095,0.01184,0.01641,0.001956,15.15,31.82,99.0,698.8,0.1162,0.1711,0.2282,0.1282,0.2871,0.06917000000000001"}

## Temizlik
*Bu lab esnasında birçok kaynak yarattınız. Daha sonra faturanıza yansımaması için:
-Sagemaker Notebook instance’ınızı kapatın
-S3 Bucket’ınızı silin
-Sagemaker Endpointinizi ve Endpoint konfigürasyonlarınızı silin
-API Gateway’de tanımlı APIlerinizi silin
-Lambda fonksiyonunuzu silin

## Authors

* **Baris Yasin** - *Initial work* - [PurpleBooth](https://github.com/barisyasin)

