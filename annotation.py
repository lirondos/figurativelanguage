#from lxml import etree
import xml.etree.ElementTree as ET
import pandas as pd
import os

class Annotation():

    def __init__(self, folder):
        self.corpus = dict()
        self.test_set = dict()
        self.training_set = dict()
        self.parse_with_stdlib(folder)
        self.split()
    """
    def parse_with_lxml(string):
        root = etree.fromstring(string)
        for log in root.xpath("//TEXT"):
            print(log.text)
    """
    def parse_with_stdlib(self, folder):
        #with open("full annotation/Molly1.txt.xml") as f:
            #string = f.read()
        for rootdir, dirs, files in os.walk(folder):
                for filename in files:
                    root = ET.parse(folder+"/"+filename).getroot()
                    #root = ElementTree.fromstring(string)
                    for cdata in root.iter('TEXT'):
                        tweets_dict = dict()
                        tweets = cdata.text.split("\n")
                        for tweet in tweets:
                            instance_list= tweet.split("\t")
                            tweet_id = instance_list[0]
                            text = instance_list[2]
                            tweets_dict[tweet_id] = text
                    tags = root.find('./TAGS')
                    for tag in tags:
                        tweet_id = tag.attrib['text']
                        label = tag.tag
                        if label == 'Figurative':
                            l = tag.attrib['type']
                            tweet = tweets_dict.get(tweet_id)
                            self.corpus[tweet_id] = (tweet, l)
                        else:
                            tweet = tweets_dict.get(tweet_id)
                            self.corpus[tweet_id] = (tweet, label)
                        


    def split(self):
        split_idx = (len(self.corpus)*80)//100 # we split the annotation into 20% test and 80% training
        self.training_set = dict(list(self.corpus.items())[:split_idx])
        self.test_set = dict(list(self.corpus.items())[split_idx:])

    def my_to_csv(self):
        with open("output.csv", 'w', encoding= 'utf-8') as f:
            f.write("id,type,tweet,label\n")
            for id, element in self.training_set.items():
                tweet = element[0].replace("\"","\"\"")
                label = element[1]
                f.write(id + ",train," + "\"" + tweet + "\"" + "," + label + "\n")
            for id, element in self.test_set.items():
                tweet = element[0].replace("\"","\"\"")
                label = element[1]
                f.write(id + ",test," + "\"" + tweet + "\"" + "," + label + "\n")
        f = pd.read_csv("output.csv", header=None)
        d = f.sample(frac = 1)
        d.to_csv('output_final.csv')
        
       
if __name__ == '__main__':
    #parse_with_lxml(string)
    annotation = Annotation("gold")
    annotation.my_to_csv()
   
    
    #print(annotation.corpus)
