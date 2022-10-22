# AI-Templates
Samples for rapid model development.

## CNN

#### traffic.py
  An assignment (week 5) from AI50.
  - Function          : Classification of road based signs via TensorFlow Keras Sequential Model.
  - Labeled Data Set  : [German Traffic Sign Recognition Benchmark](https://github.com/user/repo/blob/branch/other_file.md)
  - Dependencies      : opencv-python (for image processing), scikit-learn (ML related functions), tensorflow (for NN)<br>
    ↳ pip3 install <em>"dependency"</em>
    
    ##### Variations :
    ```
    - C1 → P1 → F → H1(32) → D(0.5) → Output
      333/333 - 1s - loss: 3.4948 - accuracy: 0.0582 - 841ms/epoch - 3ms/step
    ```
    ```
    - C1 → P1 → F → H1(64) → D(0.5) → Output
      333/333 - 1s - loss: 3.5018 - accuracy: 0.0566 - 890ms/epoch - 3ms/step
    ```
    ```
    - C1 → P1 → C2 → F → H1(64) → D(0.5) → Output
      333/333 - 1s - loss: 0.3192 - accuracy: 0.9089 - 952ms/epoch - 3ms/step
    ```
    ```
    - C1 → P1 → C2 → P2 → F → H1(128) → D(0.5) → Output
      333/333 - 1s - loss: 0.2098 - accuracy: 0.9428 - 973ms/epoch - 3ms/step
    ```
    ```
    - C1 → P1 → C2 → P2 → F → H1(256) → D(0.5) → Output
      333/333 - 1s - loss: 0.1974 - accuracy: 0.9528 - 987ms/epoch - 3ms/step
    ```
    ```
    - C1 → P1 → C2 → P2 → F → H1(256) → H2(256) → D(0.5) → Output
      333/333 - 1s - loss: 0.1230 - accuracy: 0.9697 - 1s/epoch - 3ms/step
    ```
    ```
    - C1 → P1 → C2 → P2 → H1(256) → H2(256) → F → D(0.5) → Output
      333/333 - 1s - loss: 0.1260 - accuracy: 0.9667 - 1s/epoch - 4ms/step
    ```
    ```
    - C1 → P1 → H1(256) → C2 → P2 → H2(256) → F → D(0.5) → Output
      333/333 - 2s - loss: 3.4962 - accuracy: 0.0553 - 2s/epoch - 7ms/step
    ```
    ```
    - C1 → P1 → C2 → P2 → C3 → P3 → H1(256) → F → D(0.5) → Output
      333/333 - 1s - loss: 0.1954 - accuracy: 0.9489 - 1s/epoch - 3ms/step
    ```
    ```
    - C1 → P1 → C2 → P2 → C3 → P3 → H1(256) → H2(256) → F → D(0.5) → Output
      333/333 - 1s - loss: 0.1302 - accuracy: 0.9654 - 1s/epoch - 3ms/step
    ```
    ```
    - C1 → P1 → C2 → P2 → C3 → P3 → F → H1(256) → H2(256) → D(0.5) → Output
      333/333 - 1s - loss: 0.2642 - accuracy: 0.9349 - 1s/epoch - 3ms/step
    ```
    <sub>Notation : C ← convolutional layer ,D ← droput layer, P ← maxpooling layer, H ← hidden layer, F ← flattening layer.</sub>
    
