# Titanic-Group

## Descrizione del Progetto
Questo progetto utilizza il famoso dataset Titanic di Kaggle per sviluppare modelli di machine learning in grado di prevedere quali passeggeri sono sopravvissuti al naufragio del Titanic.

## Dataset
Il dataset contiene informazioni sui passeggeri del Titanic, tra cui:
- PassengerId: ID univoco del passeggero
- Survived: Se il passeggero è sopravvissuto (1) o no (0)
- Pclass: Classe del biglietto (1, 2, 3)
- Name: Nome del passeggero
- Sex: Sesso del passeggero
- Age: Età del passeggero
- SibSp: Numero di fratelli/coniugi a bordo
- Parch: Numero di genitori/figli a bordo
- Ticket: Numero del biglietto
- Fare: Tariffa del biglietto
- Cabin: Numero della cabina
- Embarked: Porto di imbarco (C = Cherbourg, Q = Queenstown, S = Southampton)

## Analisi Effettuata
Nel nostro progetto abbiamo:

1. **Esplorazione dei dati**:
   - Analisi dei valori mancanti
   - Statistiche descrittive
   - Visualizzazione della distribuzione dei dati

2. **Preprocessing**:
   - Gestione dei valori mancanti (età, tariffa)
   - Feature engineering (creazione di 'Has_Cabin', 'FamilySize')
   - Codifica delle variabili categoriche

3. **Modellazione**:
   - Regressione Logistica
   - Random Forest
   - XGBoost

4. **Valutazione**:
   - Accuratezza
   - Matrice di confusione
   - Curva ROC
   - Importanza delle feature

## Risultati
Abbiamo confrontato le prestazioni di diversi modelli di machine learning per identificare quali caratteristiche sono più importanti per prevedere la sopravvivenza dei passeggeri. I risultati mostrano correlazioni significative tra la sopravvivenza e fattori come il sesso, la classe del biglietto e l'età.
