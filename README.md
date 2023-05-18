# CalculateGaitSpeedApp
Repozytorium kodu do projektu Aplikacji do obliczania prędkości chodu realizowanego w ramach przedmiotu Podstawy telemedycyny

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
