# Репорт и остальные части

[https://smooth-pajama-7af.notion.site/In-Context-RL-66d56cb4a4e34bfa86a9c9e289b7a5dd?pvs=4](https://smooth-pajama-7af.notion.site/In-Context-RL-66d56cb4a4e34bfa86a9c9e289b7a5dd?pvs=4)

# Структура кода

В директории /algos содержатся алгоритмы для обучения - UCB, Transformer, Random.
В директории /data содержится класс для формирования датасета для модели.
В дирктории /env содержится все про environment и его обертку.
В директории /utils содержатся мелкие утилиты, необходимые для обучения.

# Запуск кода

* Для начала загрузите все необходимые библиотеки для обучения
```
pip install -r requirements.txt
```

* Далее запустите сбор датасета
```
python3 dataset_generator.py
```

* После этого в корневой папке появится файл `trajectories.npz`, в котором и собраны данные. После этого запустите обучение AD

```
python3 train.py
```
