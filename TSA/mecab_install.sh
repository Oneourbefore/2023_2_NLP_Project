# mecab 설치
git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
cd ./Mecab-ko-for-Google-Colab
bash install_mecab-ko_on_colab_light_220429.sh
cd ./

#사용자 사전 정의 후 반영
bash autogen.sh
sudo make install
bash tools/add-userdic.sh
make clean
make install