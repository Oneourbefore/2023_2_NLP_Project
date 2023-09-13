import json
import re
import numpy as np

from kiwipiepy import Kiwi
import math
import os
import subprocess
from jamo import h2j, j2hcj

# google colab에서 실행할 경우 아래 함수 실행
def install_mecab():
    try:
        current_directory = os.getcwd()
        desired_directory = 'TSA'

        if not current_directory.endswith(desired_directory):
            print(f"Current directory {current_directory} does not match the desired directory {desired_directory}")
            os.chdir(desired_directory)

        subprocess.run(['git', 'clone', 'https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git'])
        os.chdir('Mecab-ko-for-Google-Colab')
        subprocess.run(['bash', 'install_mecab-ko_on_colab_light_220429.sh'])

    except subprocess.CalledProssError as e:
        print(f"Error executing the script: {e}")
    except Exception as e:
        print(f"An error occurred {e}")

    else :
        from konlpy.tag import Mecab  # KoNLPy를 통해 Mecab 패키지 import

def get_jongsung_TF(sample_text):
    sample_text_list = list(sample_text)
    last_word = sample_text_list[-1]
    last_word_jamo_list = list(j2hcj(h2j(last_word)))
    last_jamo = last_word_jamo_list[-1]

    jongsung_tf = "T"

    if last_jamo in ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ', 'ㅘ', 'ㅚ', 'ㅙ', 'ㅝ', 'ㅞ', 'ㅢ', 'ㅐ,ㅔ', 'ㅟ', 'ㅖ', 'ㅒ']:
        jongsung_tf = "F"

    return jongsung_tf


### mysql 연결 example
from dataset import MysqlConnection # import 구문 있는 곳(상단)에 넣으면 됨
db_connection = MysqlConnection()
conn = db_connection.connection
cursor = conn.cursor()
cursor.execute("")
result = cursor.fetchall()

cursor.close()
conn.close()