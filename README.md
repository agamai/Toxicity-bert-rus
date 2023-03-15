# Дообучение BERT на данных по токсичности

Работа представлена в двух файлах:
1. [Дообучение (ipynb)](https://github.com/agamai/Portfolio/blob/main/Toxicity_rus/my_toxicity_15ep_final.ipynb)
2. [Тестирование (ipynb)](https://github.com/agamai/Portfolio/blob/main/Toxicity_rus/my_toxicity_15ep_final_test.ipynb)

## Общее описание
Необходимо обучить модель классифицировать тексты (посты и комментарии из российских соцсетей) на предмет токсичности. Тексты были предварительно размечены вручную на наличие токсичности по нескольким категориям: не токсично, мат/нецензурное, грубое высказывание, дискриминация, оскорбление, угроза. Каждому тексту может соответствовать несколько категорий. Таким образом, задача сводится к классификации с несколькими метками (multi-label classification). Мы будем использовать предобученную модель  [*rubert-tiny-toxicity*](https://huggingface.co/cointegrated/rubert-tiny-toxicity), опубликованную на [*huggingface*](https://huggingface.co/). Имеющийся датасет будем использовать для дообучения модели на GPU.

## Инструменты
* **python**
* **pandas**
* **numpy**
* **math**
* **matplotlib**
* **tqdm**
* **re**
* **torch**
* torch.**cuda**
* torch.utils.data.**Dataset**
* sklearn.model_selection.**train_test_split**
* sklearn.metrics.**roc_curve**
* sklearn.metrics.**precision_recall_curve**
* sklearn.utils.**shuffle**
* transformers.**Trainer**
* transformers.**TrainingArguments**
* transformers.**BertTokenizer**
* transformers.**BertForSequenceClassification**
* tensorflow.keras.metrics.**CategoricalAccuracy**
* tensorflow.keras.metrics.**CategoricalCrossentropy**
* tensorflow.keras.metrics.**Precision**
* tensorflow.keras.metrics.**Recall**
* tensorflow.keras.metrics.**AUC**

## Выводы
В ходе предобработки данных были сформированы векторы меток, приемлемые для предобученой модели, был проработан дисбаланс меток. Текстовые данные были очищены, токенизированы и разделены на выборки. Для дообучения модели использовался объект класса transformers.Trainer. Дообучение проходило на GPU в течение 15 эпох, learning_rate - 1e-5. На тестовой выборке были получены метрики для дообученной и изначальной модели.

В разрезе по каждой категории токсичности дообученная модель показала лучшие результаты по всем метрикам ('accuracy', 'precision', 'recall', 'f1', 'roc_auc') по сравнению с изначальной моделью. accuracy: 0.74-0.93, f1: 0.72-0.92, precision: 0.7-0.91, recall: 0.74-0.93, roc_auc: 0.86-0.97 (в зависимости от категории). Метрики для классификации с несколькими метками также оказались лучшими у дообученной модели (все, кроме Recal): PR_auc: 0.81, Roc_auc: 0.94, Categorical accuracy: 0.76, 
Precision: 0.84; Recall: 0.8. Графики ROC-AUC и Precision/Recall показали, что дообученная модель показывает лучшие характеристики в плане TPR/FPR, а также в плане соотношения Precision/Recall. Также дообученная модель показала лучшие результаты по сравнению с изначальной по агрегированной токсичности: accuracy: 0.88, precision: 0.85, recall: 0.94, f1: 0.89, roc_auc: 0.88.
