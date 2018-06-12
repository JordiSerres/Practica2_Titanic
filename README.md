## Practica 2 Titanic
Assignatura_ Cicle de vida i tipologia de les dades - Pràctica 2

Autor: Jordi Serres

# Descripció
Participació a la competició de Kaggle: Titanic: Machine Learning from Disaster. Es tracta d'un problema de classificació binària supervisada.

S'han provat els models Decision Tree, Random Forest i Support Vector Machine del paquet sklearn. En aquest paquet els models necessiten que la informació sigui numèrica (o booleana) i que no hi hagi valors nuls. Els datasets facilitats tenien valors nuls en els camps:
* Age: S'ha assignat la mitjana a tots els valors nuls
* Cabin:S'ha creat una variable "With Cabin" que indica si el passatger tenia cabina (cas "Cabin" = Null) o no (resta de casos) i és la que s'ha utilitzat en els models.
* Fare (només el fitxer test.csv tenia nuls): S'ha assignat la mitjana a tots els valors nuls
* La variable categòrica "Embarked" pren 3 valors i també conte valors nuls. Aquesta, però, s'ha transformat en 3 variables dummy i els  registres nuls han quedat amb un 0 a les 3 variables

# Resultats
El millor resultat a Kaggle han estat les prediccions fetes amb el classificador Random Forest, aplicat a les variables ['Pclass','Sex', 'Age', 'SibSp','Parch',  'With Cabin', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S'] amb 10 estimadors.

# Millores futures
* Tractament dels valors nuls en funció d'altres variables
* Utilitzar la informació dels camps no utilitzats :'Name' i Ticket'
* Provar altres models
* Combinar models

# Paquets utilitzats:
* pandas
* numpy
* sklearn
* io


# Fitxers
* **src/titanic_kaggle.py**: Fitxer amb el codi font
* **src/train.csv**: Dataset etiquetat per entrenar i avalaur els models
* **src/test.csv**: Dataset del que s'ha de fer predicció per pujar a Kaggle
* **submission_clf_2_best.csv**: Prediccions del fitxer de test amb millor resultat obtingut a Kaggle (accuracy del 77.033%)
* **arbre_decisio_deep_4.dot**: Arbre de decisió de profunditat 4 generat. Es pot visualitzar incrustant el contingut a http://www.webgraphviz.com/

