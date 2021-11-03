# Tugas_Akhir
Gabungin dua output (1 Gambar dari ML dan 1 Jarak dari LiDAR.
Jadi Objek akan di di ukur pakai dua sensor (Kamera untuk visual 2D dan LiDAR untuk mengukur jarak antar kapal dan objek)
Output yang ingin di proses adalah
Kamera = label model ML (1 = benda terdeteksi, 0 = tidak ada benda yang terdeteksi)
LiDAR = jarak dalam CM (karena maksimal deteksi sensor hanya 2M)
setelah menerima data dari dua sensor tersebut maka ada 3 kemungkinan untuk pilihan kapal, yakni:
1. Tetap laju dengan kecepatan awal (jika tidak ada benda dan jarak tidak terdeteksi)
2. Berpelan-pelan jika ada benda dan jaraknya mendekati kapal
3. Berhenti jika Kapal dan benda sudah terlalu dekat

opsional:
pengendalian servo rudder untuk menghindari

![image](https://user-images.githubusercontent.com/92085498/140053610-b85694a7-4dfd-41e3-ba7e-e23106e5a1be.png)

