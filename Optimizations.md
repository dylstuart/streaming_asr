## Emformer Layer Architecture
<img width="597" height="514" alt="image" src="https://github.com/user-attachments/assets/1ebe15d3-ef03-40e1-99e5-6b4b34eaac94" />

## MLA
<img width="1547" height="472" alt="image" src="https://github.com/user-attachments/assets/9e837bd7-81b8-45f6-ab92-a9a65e3774c2" />
DeepSeek’s innovation is to introduce a weight matrix WDKV∈Rdc×d to compress the input X∈Rd×n to a lower rank matrix CKV∈Rdc×n. This CKV matrix is then stored in the cache. Then two other weight matrices WUK and WUV∈RdhH×dc uncompress the same CKV matrix to the key K and value V respectively. The above figure shows this visually.
