## EDLMPPI: Learning the Protein Language of Proteome-wide Protein-protein Binding Sites via Ensemble Deep Learning in an Interpretation Manner

![EDLMPPI](https://github.com/houzl3416/EDLMPPI/blob/main/EDLMPPI.png)

## Environment

```
conda install python==3.8.12
pip install tensorflow==2.4.1
pip install keras==2.4.3
pip install numpy==1.19.5
```

## Usages

#### Extract MBF

- Install dependencies

```
create a program directory
mkdir -p ../programs && cd ../program
```

- Install [SPRINT](https://github.com/lucian-ilie/SPRINT)

 ```
git clone https://github.com/lucian-ilie/SPRINT.git
git checkout DELPHI_Server
make compute_HSPs_parallel
 ```

- Install psiblast: 2.6.0+ and download the corresponding nr database. (The database used in EDLMPPI is Uniref90)

 ```
For Ubuntu:
sudo apt-get install ncbi-blast+
 ```

- Intall [hh-suite](https://github.com/soedinglab/hh-suite). The [database](http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/) used in DELPHI is uniprot20_2015_06.

- Install [GENN+ASAquick](http://mamiris.com/software.html)
- Install [ANCHOR](http://anchor.elte.hu/Downloads.php)

- Run the following code to extract MBF

  ```
  bash feature_computation/compute_features.sh $INPUT_FN
  ```

#### Extract ProtT5

- Install virtual environment

  ```
  conda create -n ProtT5 pyhton=3.7
  conda activate ProtT5
  ```

- Following the steps showing in the  **ProtT5-XL-UniRef50.ipynb**

- Notes:

  The result will be saved as **.npy**

#### Train Example

- Install virtual environment

  ```
  conda create -n PPI 
  conda activate PPI
  conda install python==3.8.12
  pip install tensorflow==2.4.1
  pip install keras==2.4.3
  pip install numpy==1.19.5
  ```

-  Download the **Train Data** at http://www.edlmppi.top:5002/download_train
- Following the steps showing in the  **/train/run.ipynb**

#### Predict Example

- Install virtual environment

  ```
  conda create -n PPI 
  conda activate PPI
  conda install python==3.8.12
  pip install tensorflow==2.4.1
  pip install keras==2.4.3
  pip install numpy==1.19.5
  ```

- The **Predict Example** can be download in current repository

  ```
  python ./predict/pred.py
  ```

#### Predict Online

- If you want to get predicted result faster, you can go to http://www.edlmppi.top:5002/ using our online predicted tool.

### Contact

- If you have any question, you can contact **houzl20@mails.jlu.edu.cn**

链接: https://pan.baidu.com/s/1UdpI5yENzCxgdd3ZSq66Ng?pwd=52jt 提取码: 52jt 复制这段内容后打开百度网盘手机App，操作更方便哦
