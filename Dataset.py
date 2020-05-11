import os
import pandas as pd
import librosa





path='D:/dev-clean/LibriSpeech/dev-clean'

def preprocessing_text(label):

    number=[]
    conv=[]
    conversation=[]
    text=''
    for a in range(len(label)):
       with open(label[a],'r') as textfile:

           line=textfile.readlines()

           for i in range(len(line)):
               sentence=line[i]

               j=0
               while(sentence[j].isdigit() or sentence[j]=='-'):

                   index=j
                   j+=1


               num=sentence[:index+1]
               word=sentence[index+1:]
               number.append(num)
               conv.append(word)

    print('preprocessing_txt complete 2')

    return number,conv


def txt_to_csv(number,conv):

    df=pd.DataFrame(columns=['number','conv'])
    a=0
    for i in range(len(number)):
            concat=pd.DataFrame({'number':number[i],'conv':conv[i]},index=[a])
            a+=1
            df=pd.concat([df,concat])

    print('txt_to_csv complete 3')
    return df



def preprocessing_number(subfolder):

    input=[]
    label=[]
    wav_file=[]
    for i in range(len(subfolder)):
        a = subfolder[i]
        folder = os.listdir(a)
        for j in range(len(folder)):
            c = os.path.join(subfolder[i], folder[j])
            wav = os.listdir(c)
            for k in range(len(wav)):
                if 'flac' in wav[k]:
                    wav_is=wav[k]
                    wav_file.append(wav_is)
                    wav_name = os.path.join(c, wav[k])
                    input.append(wav_name)
                if '.txt' in wav[k]:
                    labe = os.path.join(c, wav[k])
                    label.append(labe)


    print('preprocessing number 1 complete')

    return input,label,wav_file

def preprocessing_audio(audio):

    # first audio preprocessing:

    signal,rate=librosa.load(audio,sr=16000)

    mfcc=librosa.feature.mfcc(signal,sr=16000)

    return mfcc


def processing_all(root):

        input_pre=[]
        label_pre=[]

        mainfolder=os.listdir(root)
        subfolder=[os.path.join(root,i) for i in mainfolder]
        input,label,wav_file=preprocessing_number(subfolder)

        number,conv=preprocessing_text(label)
        pro_label=txt_to_csv(number,conv)

        for i in range(len(input)):

            split_list=input[i].split('\\')
            split=split_list[-1]

            for j in range(len(pro_label)):

                wav_name=wav_file[j]

                if split == wav_name:

                    dialog=pro_label['conv'][j]
                    audio=input[i]

                    audio=preprocessing_audio(audio)
                    input_pre.append(audio)
                    label_pre.append(dialog)

        print('audio preprocessing complete 4')

        return input_pre, label_pre






