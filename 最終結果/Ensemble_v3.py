# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 13:37:13 2019

@author: Yuichi Omozaki
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# クラス
class ResultMaster() :
    def __init__(self, _dirName, _islandNum) :
        # 各種変数

        #評価用データに対する各種誤識別率
        self.tstMissRates_Global = []
        self.tstMissRates_Local = []
        self.tstMissRates_Single = []
        #学習用データに対する各種誤識別率
        self.traMissRates_Global = []
        self.traMissRates_Local = []
        self.traMissRates_Single = []

        #弱識別器の情報
        self.Global_NonDomi = []
        self.Global_Single = []
        self.Local_NonDomi = []
        self.Local_Single = []
        self.BestSingle = []

        self.sep = '/'
        self.directory = '.' + self.sep + _dirName

        self.islandNum = _islandNum

    # 各種誤識別率 読込
    def getMissRates(self) :
        dirName = self.directory + self.sep + 'ensemble' + self.sep

        # tst ensembleGlobal
        fileName = 'tst_missRates_ensembleGlobal.txt'
        path = '' + dirName + fileName
        readMissRate(path, self.tstMissRates_Global)

        # tst ensembleLocal
        fileName = 'tst_missRates_ensembleLocal.txt'
        path = '' + dirName + fileName
        readMissRate(path, self.tstMissRates_Local)

        # tra ensembleGlobal
        fileName = 'tra_missRates_ensembleGlobal.txt'
        path = '' + dirName + fileName
        readMissRate(path, self.traMissRates_Global)

        # tra ensembleLocal
        fileName = 'tra_missRates_ensembleLocal.txt'
        path = '' + dirName + fileName
        readMissRate(path, self.traMissRates_Local)

        # tst Single
        fileName = 'tst_missRates_SingleClassifier.txt'
        path = '' + dirName + fileName
        i = 0
        f = open(path, 'r')
        for line in f:
            self.tstMissRates_Single.append([])
            self.tstMissRates_Single[i].append( line[:-1].split(' ')[0] + line[:-1].split(' ')[1]) # repeatNum + crossValidation
            self.tstMissRates_Single[i].append( line[:-1].split(' ')[2] ) # 単一識別器の評価用データに対する誤識別率
            i += 1
        f.close()
        # float変換
        for i in range(len(self.tstMissRates_Single)):
            self.tstMissRates_Single[i] =  [float(s) for s in self.tstMissRates_Single[i]]

        # tra Single
        fileName = 'tra_missRates_SingleClassifier.txt'
        path = '' + dirName + fileName
        i = 0
        f = open(path, 'r')
        for line in f:
            self.traMissRates_Single.append([])
            self.traMissRates_Single[i].append( line[:-1].split(' ')[0] + line[:-1].split(' ')[1]) # repeatNum + crossValidation
            self.traMissRates_Single[i].append( line[:-1].split(' ')[2] ) # 単一識別器の学習用データに対する誤識別率
            i += 1
        f.close()
        # float変換
        for i in range(len(self.traMissRates_Single)):
            self.traMissRates_Single[i] =  [float(s) for s in self.traMissRates_Single[i]]

    # 非劣解集合 read関数
    def readRuleSet_NonDomi(self, _fileName, _list) :
        dirName = self.directory + self.sep + 'ensemble' + self.sep + 'ruleset' + self.sep

        for i in range(30) : #繰り返し回数 repeatNum * crossValidation + crossValidation
            rr = int(i / 10)
            cc = i % 10
            fileName = _fileName
            fileName += self.sep + 'rules_' + str(rr) + '_' + str(cc) + '.txt'
            path = '' + dirName + fileName
            _list.append([]) # rr ccの個数 list を生成

            f = open(path, 'r')
            flg = 0
            j = -1
            for line in f :
                if '---' in line :
                    _list[i].append([])
                    ruleNum = 0
                    ruleset_i = 0
                    flg = 0
                    j += 1
                elif flg == 0 :
                    _list[i][j].append({})
                    _list[i][j][0].setdefault( "island_i", int(line[:-1].split(' ')[0]))
                    _list[i][j][0].setdefault( "basic_i", int(line[:-1].split(' ')[1]))
                    _list[i][j][0].setdefault( "fitness0", float(line[:-1].split(' ')[2]))
                    _list[i][j][0].setdefault( "fitness1", float(line[:-1].split(' ')[3]))
                    ruleNum = float(line[:-1].split(' ')[3])
                    Ndim = float(line[:-1].split(' ')[4])
                    flg = 1
                else :
                    _list[i][j].append({})
                    _list[i][j][ruleset_i + 1].setdefault("rule", [])
                    for rule_i in range(int(Ndim)) :
                        _list[i][j][ruleset_i + 1]["rule"].append(int(line[:-1].split(' ')[rule_i]))
                    _list[i][j][ruleset_i + 1].setdefault("conClass", int(line[:-1].split(' ')[int(Ndim)]))
                    _list[i][j][ruleset_i + 1].setdefault("Cf", line[:-1].split(' ')[int(Ndim) + 1])
                    ruleset_i += 1

    # 単一弱識別器 read関数
    def readRuleSet_Single(self, _fileName, _list) :
        dirName = self.directory + self.sep + 'ensemble' + self.sep + 'ruleset' + self.sep

        for i in range(30) : #繰り返し回数 repeatNum * crossValidation + crossValidation
            rr = int(i / 10)
            cc = i % 10
            fileName = _fileName
            fileName += self.sep + 'rules_' + str(rr) + '_' + str(cc) + '.txt'
            path = '' + dirName + fileName
            _list.append([]) # rr ccの個数 list を生成

            f = open(path, 'r')
            flg = 0
            j = -1
            for line in f :
                if '---' in line :
                    _list[i].append([])
                    ruleNum = 0
                    ruleset_i = 0
                    flg = 0
                    j += 1
                elif flg == 0 :
                    _list[i][j].append({})
                    _list[i][j][0].setdefault( "island_i", int(line[:-1].split(' ')[0]))
                    _list[i][j][0].setdefault( "fitness0", float(line[:-1].split(' ')[1]))
                    _list[i][j][0].setdefault( "fitness1", float(line[:-1].split(' ')[2]))
                    ruleNum = float(line[:-1].split(' ')[2])
                    Ndim = float(line[:-1].split(' ')[3])
                    flg = 1
                else :
                    _list[i][j].append({})
                    _list[i][j][ruleset_i + 1].setdefault("rule", [])
                    for rule_i in range(int(Ndim)) :
                        _list[i][j][ruleset_i + 1]["rule"].append(int(line[:-1].split(' ')[rule_i]))
                    _list[i][j][ruleset_i + 1].setdefault("conClass", int(line[:-1].split(' ')[int(Ndim)]))
                    _list[i][j][ruleset_i + 1].setdefault("Cf", line[:-1].split(' ')[int(Ndim) + 1])
                    ruleset_i += 1

    # best単一識別器 read関数
    def readRuleSet_BestSingle(self, _fileName, _list) :
        dirName = self.directory + self.sep + 'ensemble' + self.sep + 'ruleset' + self.sep

        for i in range(30) : #繰り返し回数 repeatNum * crossValidation + crossValidation
            rr = int(i / 10)
            cc = i % 10
            fileName = _fileName
            fileName += self.sep + 'rules_' + str(rr) + '_' + str(cc) + '.txt'
            path = '' + dirName + fileName
            _list.append([]) # rr ccの個数 list を生成

            f = open(path, 'r')
            flg = 0
            for line in f :
                if '---' in line :
                    ruleNum = 0
                    ruleset_i = 0
                    flg = 0
                elif flg == 0 :
                    _list[i].append({})
                    _list[i][0].setdefault( "fitness0", float(line[:-1].split(' ')[0]))
                    _list[i][0].setdefault( "fitness1", float(line[:-1].split(' ')[1]))
                    ruleNum = float(line[:-1].split(' ')[1])
                    Ndim = float(line[:-1].split(' ')[2])
                    flg = 1
                else :
                    _list[i].append({})
                    _list[i][ruleset_i + 1].setdefault("rule", [])
                    for rule_i in range(int(Ndim)) :
                        _list[i][ruleset_i + 1]["rule"].append(int(line[:-1].split(' ')[rule_i]))
                    _list[i][ruleset_i + 1].setdefault("conClass", int(line[:-1].split(' ')[int(Ndim)]))
                    _list[i][ruleset_i + 1].setdefault("Cf", line[:-1].split(' ')[int(Ndim) + 1])
                    ruleset_i += 1

    # ピッツバーグ個体群 (識別器) 情報 読込
    def getRuleSet(self) :
        self.readRuleSet_NonDomi('Global_NonDomi', self.Global_NonDomi)
        self.readRuleSet_NonDomi("Local_NonDomi" , self.Local_NonDomi)
        self.readRuleSet_Single('Global_Single', self.Global_Single)
        self.readRuleSet_Single('Local_Single', self.Local_Single)
        self.readRuleSet_BestSingle('SingleBestClassifier', self.BestSingle)

    # 多目的空間における全ての弱識別器の可視化
    #     各CVの弱識別器集合を引数として与える
    def showNonDomiAllPop(self, _list) :
        fig_nonDomi = plt.figure(figsize=(5, 5))
        axes_0 = fig_nonDomi.add_subplot(111)
#         axes_0.set_title("NonDomination All Basic Classifiers")

        fitnessX = []
        fitnessY = []
        for basic in _list :
            fitnessX.append(basic[0]["fitness1"])
            fitnessY.append(basic[0]["fitness0"])
        popNum = len(fitnessX)
        axes_0.scatter(fitnessX, fitnessY, s=100, linewidths=1, edgecolors='black', label="popSize : " + str(popNum))

        axes_0.set_ylabel("Error Rate [%]")
        axes_0.set_xlabel("The Number of Rules [-]")
        axes_0.legend()
        axes_0.set_xlim([0, 30])
        axes_0.set_ylim([0.0, 40.0])
        axes_0.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        axes_0.tick_params(axis='both', which='major', length=6)
        
        return fig_nonDomi

    # 多目的空間における各島の弱識別器の可視化
    #     各CVの弱識別器集合を引数として与える
    def showNonDomiIslandPop(self, _list, xmin, xmax, ymin, ymax) :
        fig_nonDomi = plt.figure(figsize=(5, 5))
        axes_0 = fig_nonDomi.add_subplot(111)
#         axes_0.set_title("NonDomination Basic Classifiers from Each Islands")

        for i in range(self.islandNum) :
            fitnessX = []
            fitnessY = []
            missRateX = []
            missRateY = []
            for basic in _list :
                if basic[0]["island_i"] == i :
                    fitnessX.append(basic[0]["fitness1"])
                    fitnessY.append(basic[0]["fitness0"])
            popNum = len(fitnessX)
            axes_0.scatter(fitnessX, fitnessY, s=100, linewidths=1, edgecolors='black', label="island" + str(i) + " : " + str(popNum))

        axes_0.set_ylabel("Error Rate", fontsize=16)
        axes_0.set_xlabel("The Number of Rules", fontsize=16)
        axes_0.legend(fontsize=16, prop={'size': 12,})
        axes_0.set_xlim([xmin, xmax])
        axes_0.set_ylim([ymin, ymax])
        axes_0.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        axes_0.tick_params(axis='both', which='major', length=6)
        
        return fig_nonDomi
        
    # 多目的空間における指定した島の弱識別器の可視化
    #     各CVの弱識別器集合を引数として与える
    def showNonDomiEspeciallyIslandPop(self, _list, _island_i) :
        fig_nonDomi = plt.figure(figsize=(5, 5))
        axes_0 = fig_nonDomi.add_subplot(111)
#         axes_0.set_title("NonDomination Basic Classifiers from Each Islands")

        
        fitnessX = []
        fitnessY = []
        for basic in _list :
             if basic[0]["island_i"] == _island_i :
                fitnessX.append(basic[0]["fitness1"])
                fitnessY.append(basic[0]["fitness0"])
        popNum = len(fitnessX)
        axes_0.scatter(fitnessX, fitnessY, s=100, linewidths=1, edgecolors='black', label="island" + str(_island_i) + " : " + str(popNum))

        axes_0.set_ylabel("Error Rate")
        axes_0.set_xlabel("The Number of Rules")
        axes_0.legend()
        axes_0.set_xlim([0, 30])
        axes_0.set_ylim([10.0, 40.0])
        axes_0.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        axes_0.tick_params(axis='both', which='major', length=6)
        
        return fig_nonDomi

# 誤識別率 read関数
def readMissRate(_path, _list) :
    i = 0
    f = open(_path, 'r')
    for line in f:
        _list.append([])
        _list[i].append( line[:-1].split(' ')[0]) # repeatNum
        _list[i].append( line[:-1].split(' ')[1]) # crossValidation
        _list[i].append( line[:-1].split(' ')[2]) # 単一な弱識別器 単純多数決
        _list[i].append( line[:-1].split(' ')[3]) # 単一な弱識別器 サブデータ重み多数決
        _list[i].append( line[:-1].split(' ')[4]) # 単一な弱識別器 全データ重み多数決
        _list[i].append( line[:-1].split(' ')[5]) # 非劣解集合 島統合 単純多数決
        _list[i].append( line[:-1].split(' ')[6]) # 非劣解集合 島統合 サブデータ重み多数決
        _list[i].append( line[:-1].split(' ')[7]) # 非劣解集合 島統合 全データ重み多数決
        _list[i].append( line[:-1].split(' ')[8]) # 非劣解集合 島ごと個別 単純多数決
        _list[i].append( line[:-1].split(' ')[9]) # 非劣解集合 島ごと個別 サブデータ重み多数決
        _list[i].append( line[:-1].split(' ')[10]) # 非劣解集合 島ごと個別 全データ重み多数決
        i += 1
    f.close()
    # float変換
    for i in range(len(_list)):
        _list[i] =  [float(s) for s in _list[i]]


def averageFitnessSingle(_list):
    ave = 0
    for i in range(30):
        ave += _list[i][0]["fitness0"]
    ave /= 30
    print(100 - ave)

def showNonDomiIslandPop(_instance, _tryNum, xmin, xmax, ymin, ymax) :
    fig_nonDomi = plt.figure(figsize=(5, 5))
    axes_0 = fig_nonDomi.add_subplot(111)
    _list = _instance.Global_NonDomi[_tryNum]
#         axes_0.set_title("NonDomination Basic Classifiers from Each Islands")

    for i in range(_instance.islandNum) :
        fitnessX = []
        fitnessY = []
        missRateX = []
        missRateY = []
        for basic in _list :
            if basic[0]["island_i"] == i :
                fitnessX.append(basic[0]["fitness1"])
                fitnessY.append(basic[0]["fitness0"])
        popNum = len(fitnessX)
#        axes_0.scatter(fitnessX, fitnessY, s=100, linewidths=1, edgecolors='black', label="island" + str(i) + " : " + str(popNum))
        axes_0.scatter(fitnessX, fitnessY, s=100, linewidths=1, edgecolors='black', label="island" + str(i))

    axes_0.set_ylabel("Error Rate", fontsize=16)
    axes_0.set_xlabel("The Number of Rules", fontsize=16)
    axes_0.legend(prop={'size':16}, fontsize=12)
    axes_0.set_xlim([xmin, xmax])
    axes_0.set_ylim([ymin, ymax])
    axes_0.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    axes_0.tick_params(axis='both', which='major', length=6)
    axes_0.tick_params(labelsize=16)
        
    return fig_nonDomi    



# main

# 各種インスタンス
overFit_3_phoneme = ResultMaster('./overFit_island3_phoneme', 3)
overFit_3_satimage = ResultMaster('./overFit_island3_satimage', 3)
overFit_5_phoneme = ResultMaster('./overFit_island5_phoneme', 5)
overFit_5_satimage = ResultMaster('./overFit_island5_satimage', 5)
overFit_7_phoneme = ResultMaster('./overFit_island7_phoneme', 7)
overFit_7_satimage = ResultMaster('./overFit_island7_satimage', 7)
overFit_9_phoneme = ResultMaster('./overFit_island9_phoneme', 9)
overFit_9_satimage = ResultMaster('./overFit_island9_satimage', 9)

interval50_3_phoneme = ResultMaster('./interval50_island3_phoneme', 3)
interval50_3_satimage = ResultMaster('./interval50_island3_satimage', 3)
interval50_5_phoneme = ResultMaster('./interval50_island5_phoneme', 5)
interval50_5_satimage = ResultMaster('./interval50_island5_satimage', 5)
interval50_7_phoneme = ResultMaster('./interval50_island7_phoneme', 7)
interval50_7_satimage = ResultMaster('./interval50_island7_satimage', 7)
interval50_9_phoneme = ResultMaster('./interval50_island9_phoneme', 9)
interval50_9_satimage = ResultMaster('./interval50_island9_satimage', 9)

# 各種インスタンス初期化
overFit_3_phoneme.getMissRates()
overFit_3_phoneme.getRuleSet()
overFit_3_satimage.getMissRates()
overFit_3_satimage.getRuleSet()
print('overFit_island3 ok.')

overFit_5_phoneme.getMissRates()
overFit_5_phoneme.getRuleSet()
overFit_5_satimage.getMissRates()
overFit_5_satimage.getRuleSet()
print('overFit_island5 ok.')

overFit_7_phoneme.getMissRates()
overFit_7_phoneme.getRuleSet()
overFit_7_satimage.getMissRates()
overFit_7_satimage.getRuleSet()
print('overFit_island7 ok.')

overFit_9_phoneme.getMissRates()
overFit_9_phoneme.getRuleSet()
overFit_9_satimage.getMissRates()
overFit_9_satimage.getRuleSet()
print('overFit_island9 ok.')

interval50_3_phoneme.getMissRates()
interval50_3_phoneme.getRuleSet()
interval50_3_satimage.getMissRates()
interval50_3_satimage.getRuleSet()
print('interval50_island3 ok.')

interval50_5_phoneme.getMissRates()
interval50_5_phoneme.getRuleSet()
interval50_5_satimage.getMissRates()
interval50_5_satimage.getRuleSet()
print('interval50_island5 ok.')

interval50_7_phoneme.getMissRates()
interval50_7_phoneme.getRuleSet()
interval50_7_satimage.getMissRates()
interval50_7_satimage.getRuleSet()
print('interval50_island7 ok.')

interval50_9_phoneme.getMissRates()
interval50_9_phoneme.getRuleSet()
interval50_9_satimage.getMissRates()
interval50_9_satimage.getRuleSet()
print('interval50_island9 ok.')

# 実際の処理
# i = 1
# fig = a.showNonDomiIslandPop(a.Global_NonDomi[i])
# fig = a.showNonDomiIslandPop(a.Local_NonDomi[i])
#
#fig = overFit_9_phoneme.showNonDomiIslandPop(overFit_9_phoneme.Global_Single[0])
#fig.savefig('fig2\island9_overFit_phoneme_Global_Single_0')
#fig = overFit_9_satimage.showNonDomiIslandPop(overFit_9_satimage.Global_Single[0])
#fig.savefig('fig2\island9_overFit_satimage_Global_Single_0')
#fig = interval50_9_phoneme.showNonDomiIslandPop(interval50_9_phoneme.Global_Single[0])
#fig.savefig('fig2\island9_interval50_phoneme_Global_Single_0')
#fig = interval50_9_satimage.showNonDomiIslandPop(interval50_9_satimage.Global_Single[0])
#fig.savefig('fig2\island9_interval50_satimage_Global_Single_0')
#
#fig = overFit_7_phoneme.showNonDomiIslandPop(overFit_7_phoneme.Global_Single[0])
#fig.savefig('fig2\island7_overFit_phoneme_Global_Single_0')
#fig = overFit_7_satimage.showNonDomiIslandPop(overFit_7_satimage.Global_Single[0])
#fig.savefig('fig2\island7_overFit_satimage_Global_Single_0')
#fig = interval50_7_phoneme.showNonDomiIslandPop(interval50_7_phoneme.Global_Single[0])
#fig.savefig('fig2\island7_interval50_phoneme_Global_Single_0')
#fig = interval50_7_satimage.showNonDomiIslandPop(interval50_7_satimage.Global_Single[0])
#fig.savefig('fig2\island7_interval50_satimage_Global_Single_0')
#
#fig = overFit_5_phoneme.showNonDomiIslandPop(overFit_5_phoneme.Global_Single[0])
#fig.savefig('fig2\island5_overFit_phoneme_Global_Single_0')
#fig = overFit_5_satimage.showNonDomiIslandPop(overFit_5_satimage.Global_Single[0])
#fig.savefig('fig2\island5_overFit_satimage_Global_Single_0')
#fig = interval50_5_phoneme.showNonDomiIslandPop(interval50_5_phoneme.Global_Single[0])
#fig.savefig('fig2\island5_interval50_phoneme_Global_Single_0')
#fig = interval50_5_satimage.showNonDomiIslandPop(interval50_5_satimage.Global_Single[0])
#fig.savefig('fig2\island5_interval50_satimage_Global_Single_0')
#
#fig = overFit_3_phoneme.showNonDomiIslandPop(overFit_3_phoneme.Global_Single[0])
#fig.savefig('fig2\island3_overFit_phoneme_Global_Single_0')
#fig = overFit_3_satimage.showNonDomiIslandPop(overFit_3_satimage.Global_Single[0])
#fig.savefig('fig2\island3_overFit_satimage_Global_Single_0')
#fig = interval50_3_phoneme.showNonDomiIslandPop(interval50_3_phoneme.Global_Single[0])
#fig.savefig('fig2\island3_interval50_phoneme_Global_Single_0')
#fig = interval50_3_satimage.showNonDomiIslandPop(interval50_3_satimage.Global_Single[0])
#fig.savefig('fig2\island3_interval50_satimage_Global_Single_0')

