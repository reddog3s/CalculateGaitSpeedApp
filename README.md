# CalculateGaitSpeedApp
Repozytorium kodu do projektu Aplikacji do obliczania prędkości chodu realizowanego w ramach przedmiotu Podstawy telemedycyny. </br>
Aplikacja pozwala na obliczenie liczby kroków i prędkości chodu użytkownika korzystając z danych akcelerometrycznych i orientacji smartfona w przestrzeni.

# Wymagania
* numpy
* pandas
* SciPy
* matlplotlib
* PySimpleGUI

# Instrukcja
gui.py - aplikacja z interfejsem graficznym </br>
evaluate_all.py - plik pozwala na wczytanie wszystkich danych testowych i referencyjnych oraz ich ewaluację. </br>
Zwracana jest średnia, odchylenie standardowe, korelacja Pearsona oraz wykres Blanda-Altmana. </br>
evaluate_batch.py - pozwala na wczytanie jednej partii danych wejściowych oraz danych referencyjnych np. tylko dla chodu wolnego. </br>
Ewaluacja przebiega tak samo, jak w pliku evaluate_all.py. </br>
myUtils.py - plik zawiera bibliotekę funkcji używanych przez inne pliki projektu. </br>
VisualizeSteps.py - plik odpowiedzialny za wizualizację pośrednich kroków wykonywania algorytmu. </br>

# Źródła
Algorytm został opracowany na podstawie: </br>
P. Silsupadol, P. Prupetkaew, T. Kamnardsiri and V. Lugade, "Smartphone-Based Assessment of Gait During Straight Walking, Turning, and Walking Speed Modulation in Laboratory and Free-Living Environments," in IEEE Journal of Biomedical and Health Informatics, vol. 24, no. 4, pp. 1188-1195, April 2020, doi: 10.1109/JBHI.2019.2930091. </br>
Analiza Blanda-Altmana została wykonana na podstawie: https://rowannicholls.github.io/python/statistics/agreement/bland_altman.html </br>
Pozostałe źródła są podane w sprawozdaniu z realizacji projektu.

# [ENG]
Code repository for Calculate Gait Speed App project created for the "Basics of telemedicine" university course. </br>
App is able to calculate number of steps and gait speed of the user using accelerometer and orientation data.

# Requirements
* numpy
* pandas
* SciPy
* matlplotlib
* PySimpleGUI

# Manual
gui.py - app with it's own gui </br>
evaluate_all.py - file for evaluation of test data againts reference data. </br>
It returns mean, std, Pearson correlation and Bland-Altman plot. </br>
evaluate_batch.py - file for evaluation batch of the data, for example only for "slow gait" batch. </br>
Evaluation process is the same as in the evaluate_all.py. </br>
myUtils.py - file with functions' library used in the project</br>
VisualizeSteps.py - file responsible for visualization of intermediate steps of the algorithm. </br>

# Sources
Algorithm was based on solution proposed in: </br>
P. Silsupadol, P. Prupetkaew, T. Kamnardsiri and V. Lugade, "Smartphone-Based Assessment of Gait During Straight Walking, Turning, and Walking Speed Modulation in Laboratory and Free-Living Environments," in IEEE Journal of Biomedical and Health Informatics, vol. 24, no. 4, pp. 1188-1195, April 2020, doi: 10.1109/JBHI.2019.2930091. </br>
Bland-Altman analysis was based on: https://rowannicholls.github.io/python/statistics/agreement/bland_altman.html </br>
Other sources are available in report from project's realization.
