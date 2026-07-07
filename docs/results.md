HUMAN v1.1 go_min_count=20 MLP预测
验证集：
bp: best_epoch:6 AUC:0.9077133630530935 best_Fmax:0.40222643272396197 AUPR:0.3904520584518668 thresh:0.6742
cc: best_epoch:6 AUC:0.9466094720147156 best_Fmax:0.6675173529301833 AUPR:0.6989451389838237 thresh:0.8441
mf: best_epoch:8 AUC:0.9677573779247759 best_Fmax:0.7751251956125768 AUPR:0.799460335698982 thresh:0.8104
测试集：
bp: AUC: 0.9153 AUPR: 0.4049 Fmax: 0.4102
cc: AUC: 0.9470 AUPR: 0.6941 Fmax: 0.6583
mf: AUC: 0.9774 AUPR: 0.8209 Fmax: 0.7902

HUMAN v1.1 20 点积预测 ↓ 
验证集：
bp: best_epoch:7 AUC:0.8912371813386207 best_Fmax:0.3914730162065735 AUPR:0.3683982252305752 thresh:0.6264
cc: best_epoch:7 AUC:0.9372631363990321 best_Fmax:0.6538534535214002 AUPR:0.6820723377788376 thresh:0.8009
mf: best_epoch:9 AUC:0.9589059857438995 best_Fmax:0.7352775884583699 AUPR:0.7550640415968286 thresh:0.7502

HUMAN v2 20 MLP预测
验证集：
bp: best_epoch:7 AUC:0.9148788408692163 best_Fmax:0.3522848116220287 AUPR:0.34216167642646333 thresh:0.6047
cc: best_epoch:9 AUC:0.9649480672337862 best_Fmax:0.6493209385703109 AUPR:0.7119772121939821 thresh:0.8284
mf: best_epoch:9 AUC:0.9754048914414781 best_Fmax:0.7175144185170916 AUPR:0.7532765738897392 thresh:0.7865
测试集：
bp: AUC: 0.9133 AUPR: 0.3391 Fmax: 0.3502
cc: AUC: 0.9678 AUPR: 0.7304 Fmax: 0.6642
mf: AUC: 0.9744 AUPR: 0.7426 Fmax: 0.7067

HUMAN v2 20 点积预测 ↓ 
验证集：
bp:best_epoch:5 AUC:0.9038002575515037 best_Fmax:0.35332185056000825 AUPR:0.33567344801019355 thresh:0.5440

因为点积预测效果太差了，后面都使用MLP预测

HUMAN v1.1 20 调整MLP hid_dim=256, 直接使用sum池化+简单线性层处理seq
验证集：
bp: best_epoch:7 AUC:0.9225162573520216 best_Fmax:0.3858781332146124 AUPR:0.38212953078663875 thresh:0.6614
cc: best_epoch:8 AUC:0.9678734936499862 best_Fmax:0.6900338095325447 AUPR:0.7469207399231288 thresh:0.8602
mf: best_epoch:8 AUC:0.9774649774440975 best_Fmax:0.7507434801511118 AUPR:0.7925216459897109 thresh:0.8192
测试集：
bp: AUC: 0.9263 AUPR: 0.4000 F1_score: 0.4030
cc: AUC: 0.9682 AUPR: 0.7469 F1_score: 0.6881
mf: AUC: 0.9810 AUPR: 0.8013 F1_score: 0.7596

HUMAN v1.1 20 调整MLP hid_dim=256, 直接使用sum池化+简单线性层处理seq，不进行GO标签预训练
验证集：
bp: best_epoch:7 AUC:0.9262569719663605 best_Fmax:0.3972101036416968 AUPR:0.3965900868826659 thresh:0.6415
cc: best_epoch:9 AUC:0.9643472138722053 best_Fmax:0.6769988583093219 AUPR:0.7261569517601907 thresh:0.8854
mf: best_epoch:9 AUC:0.9775819842790824 best_Fmax:0.7735752830987785 AUPR:0.8075898100055867 thresh:0.8238
测试集：
bp: AUC: 0.9267 AUPR: 0.4045 F1_score: 0.3993
cc: AUC: 0.9654 AUPR: 0.7346 F1_score: 0.6747
mf: AUC: 0.9834 AUPR: 0.8176 F1_score: 0.7763

CAFA3 v1.1 20 调整MLP hid_dim=256, 直接使用sum池化+简单线性层处理seq
验证集：
bp: best_epoch:4 AUC:0.9532107856303897 best_Fmax:0.41382657734578565 AUPR:0.38744435949354444 thresh:0.4429
cc: best_epoch:3 AUC:0.9810560375918693 best_Fmax:0.6761002967873325 AUPR:0.7409801492875197 thresh:0.7262
mf: best_epoch:9 AUC:0.9777552938975046 best_Fmax:0.6447809452469181 AUPR:0.6879510920958993 thresh:0.7113
测试集：
bp: AUC: 0.9506 AUPR: 0.3896 F1_score: 0.4155
cc: AUC: 0.9806 AUPR: 0.7305 F1_score: 0.6602
mf: AUC: 0.9797 AUPR: 0.6805 F1_score: 0.6348

CAFA3(large) v1.1 1 调整MLP hid_dim=256, 直接使用sum池化+简单线性层处理seq
验证集：
bp: best_epoch:9 AUC:0.9663025324914988 best_Fmax:0.41293064531770124 AUPR:0.3798471862344832 thresh:0.3687
cc: best_epoch:4 AUC:0.9885468910478088 best_Fmax:0.6721805065255386 AUPR:0.7347208149586446 thresh:0.6319
mf: best_epoch:7 AUC:0.9724533151952223 best_Fmax:0.6076739986868567 AUPR:0.6241848392520107 thresh:0.5135
测试集：
bp: AUC: 0.9659 AUPR: 0.3828 F1_score: 0.4153
cc: AUC: 0.9886 AUPR: 0.7375 F1_score: 0.6752
mf: AUC: 0.9749 AUPR: 0.6449 F1_score: 0.6264


10500 v1.1 13 调整MLP hid_dim=256, 直接使用sum池化+简单线性层处理seq

bp标签1374个，cc标签234个，mf标签485个

验证集：
bp: best_epoch:4 AUC:0.9900799537071299 best_Fmax:0.8233195594836391 AUPR:0.882085002565414 thresh:0.9477
cc: best_epoch:8 AUC:0.9925121313871565 best_Fmax:0.865654200627869 AUPR:0.9177350218976714 thresh:0.8945
mf: best_epoch:8 AUC:0.9988958881134299 best_Fmax:0.9663912480291751 AUPR:0.9872653813364959 thresh:0.8854
测试集：
bp: AUC: 0.9922 AUPR: 0.9138 F1_score: 0.8656
cc: AUC: 0.9944 AUPR: 0.9174 F1_score: 0.8611
mf: AUC: 0.9980 AUPR: 0.9755 F1_score: 0.9589


10500 v1.1 1 调整MLP hid_dim=256, 直接使用sum池化+简单线性层处理seq

bp标签5756个，cc标签825个，mf标签1754个

验证集：
bp: best_epoch:9 AUC:0.9852256694681427 best_Fmax:0.7442790441766607 AUPR:0.8055715058180596 thresh:0.9346
cc: best_epoch:9 AUC:0.9928921032399436 best_Fmax:0.850787396630883 AUPR:0.8960551525051685 thresh:0.8089
mf: best_epoch:9 AUC:0.9961099956934973 best_Fmax:0.9544328825721232 AUPR:0.9748300935628917 thresh:0.7909
测试集：
bp: AUC: 0.9870 AUPR: 0.8697 F1_score: 0.8445
cc: AUC: 0.9924 AUPR: 0.8976 F1_score: 0.8487
mf: AUC: 0.9945 AUPR: 0.9652 F1_score: 0.9463


HUMAN v1.1 20 改为GVP论文一致的建图方式，KNN=30
验证集：
bp: best_epoch:17 AUC:0.9203165209858479 best_Fmax:0.3921525985297806 AUPR:0.38859204191920205 thresh:0.6631436944007874
cc: best_epoch:19 AUC:0.964619147477839 best_Fmax:0.6761749877045746 AUPR:0.7248152091867128 thresh:0.857273519039154
mf: best_epoch:18 AUC:0.9819585373328991 best_Fmax:0.773895910497146 AUPR:0.8079999469380993 thresh:0.822839081287384
测试集：
bp: AUC: 0.9292 AUPR: 0.3911 F1_score: 0.3861
cc: AUC: 0.9625 AUPR: 0.7214 F1_score: 0.6734
mf: AUC: 0.9822 AUPR: 0.8205 F1_score: 0.7852
变化不大


HUMAN v1.1 20 GVP建图+节点特征blosum62+对输出进行LayerNorm，KNN=20
验证集：
bp: best_epoch:19 AUC:0.9042906643285226 best_Fmax:0.43446961204487655 AUPR:0.4156600766312949 thresh:0.732752799987793
cc: best_epoch:18 AUC:0.9466846796505886 best_Fmax:0.6770814916688509 AUPR:0.6960729204536699 thresh:0.8590723872184753
mf: best_epoch:16 AUC:0.9759978982987694 best_Fmax:0.7949740676392719 AUPR:0.8177274180611975 thresh:0.821505606174469
测试集：
bp: AUC: 0.9132 AUPR: 0.4143 F1_score: 0.4276
cc: AUC: 0.9501 AUPR: 0.6892 F1_score: 0.6620
mf: AUC: 0.9744 AUPR: 0.8229 F1_score: 0.8036
有微弱提升，但消融测试结果局部特征和结构特征无用

HUMAN v1.1 20 GVP建图+节点特征blosum62+对输出进行LayerNorm，KNN=30
验证集：
bp: best_epoch:17 AUC:0.9051493135525496 best_Fmax:0.43518372590617177 AUPR:0.4166199377859565 thresh:0.7397469878196716
cc: best_epoch:17 AUC:0.9431214456453101 best_Fmax:0.6740165506450227 AUPR:0.690995749722294 thresh:0.8914470672607422
mf: best_epoch:16 AUC:0.9759215421031409 best_Fmax:0.7958515783988148 AUPR:0.8177609978222484 thresh:0.8584345579147339
测试集：
bp: AUC: 0.9148 AUPR: 0.4151 F1_score: 0.4281
cc: AUC: 0.9495 AUPR: 0.6866 F1_score: 0.6624
mf: AUC: 0.9742 AUPR: 0.8253 F1_score: 0.8060
有微弱提升，但消融测试结果局部特征和结构特征无用

GO:
AF2DB v1.1 20 GVP建图+节点特征blosum62+对输出进行LayerNorm，KNN=30
all:AUC: 0.9812 AUPR: 0.6400 F1_score: 0.6016
bp: AUC: 0.9719 AUPR: 0.5055 F1_score: 0.4957
cc: AUC: 0.9894 AUPR: 0.8204 F1_score: 0.7554
mf: AUC: 0.9922 AUPR: 0.8881 F1_score: 0.8427

AF2DB v3 20 GVP建图+节点特征blosum62+对输出进行LayerNorm，KNN=30
all:AUC: 0.9803 AUPR: 0.6382 Fmax: 0.5991 tmp_thresh: 0.8093
bp: AUC: 0.9708 AUPR: 0.5025 Fmax: 0.4919 tmp_thresh: 0.7926
cc: AUC: 0.9883 AUPR: 0.8178 Fmax: 0.7538 tmp_thresh: 0.8738
mf: AUC: 0.9924 AUPR: 0.8871 Fmax: 0.8413 tmp_thresh: 0.8671
使用交叉注意力之后，结果略微下降

AF2DB GO正式结果：还是用v1.1， 不筛除标签
v1.1 1 GVP建图+节点特征blosum62+对输出进行LayerNorm，KNN=30
bp: AUC: 0.9811 AUPR: 0.5090 PRC: 0.5987 REC: 0.4321  Fmax: 0.5019 tmp_thresh: 0.7454
cc: AUC: 0.9938 AUPR: 0.8203 PRC: 0.8048 REC: 0.6948  Fmax: 0.7457 tmp_thresh: 0.8053
mf: AUC: 0.9888 AUPR: 0.8686 PRC: 0.8907 REC: 0.7598  Fmax: 0.8201 tmp_thresh: 0.7632


7120 PPI测试集:
pneg=0.0 rneg=1.0 alpha1=0.5 alpha2=0.5
AUC: 0.9138 AUPR: 0.9065 Fmax: 0.8418 thresh: 0.4198
pneg=0.8 rneg=0.2 alpha1=0.5 alpha2=0.5
AUC: 0.9606 AUPR: 0.9491 Fmax: 0.9236 thresh: 0.3307

pneg=5.0 rneg=0.0 alpha1=0.5 alpha2=0.5
AUC: 0.9887 AUPR: 0.9740 Fmax: 0.9356 thresh: 0.7435



7120 GO正式结果：
直接从7120训练+预测，bp、cc、mf（最好，泛化性也很好）
AUC: 0.9237 AUPR: 0.5226 PRC: 0.5808 REC: 0.4790  Fmax: 0.5250 tmp_thresh: 0.8200
Real Average precision: 0.281454 , Real Average recall: 0.703721 , Average New GO: 6.222581

AUC: 0.9557 AUPR: 0.8283 PRC: 0.8677 REC: 0.7005  Fmax: 0.7752 tmp_thresh: 0.9300
Real Average precision: 0.339481 , Real Average recall: 0.935415 , Average New GO: 3.125489

AUC: 0.9620 AUPR: 0.7425 PRC: 0.8235 REC: 0.6418  Fmax: 0.7213 tmp_thresh: 0.8400
Real Average precision: 0.441963 , Real Average recall: 0.674027 , Average New GO: 2.565380

调整参数，观察具体标签的预测情况：
bp: Real Average precision: 0.181699 , Real Average recall: 0.817544 , Average New GO: 18.314572
cc: Real Average precision: 0.316304 , Real Average recall: 0.968716 , Average New GO: 4.406223
mf: Real Average precision: 0.301699 , Real Average recall: 0.819668 , Average New GO: 7.595137

-----------------------------------------
从AF2DB large训练+7120微调预测，微调数据为1000个蛋白质
best_epoch:0 AUC:0.9703014214939635 best_Fmax:0.4621064864526877 AUPR:0.4379017678115721 thresh:0.6833388209342957
Real Average precision: 0.304747 , Real Average recall: 0.276270 , Average New GO: 1.167829

best_epoch:9 AUC:0.9847369600257186 best_Fmax:0.7166738055452239 AUPR:0.7396969797001431 thresh:0.6855855584144592
Real Average precision: 0.290776 , Real Average recall: 0.786776 , Average New GO: 2.565890

best_epoch:9 AUC:0.9601487448575706 best_Fmax:0.48463003181116354 AUPR:0.48175939490096814 thresh:0.6112263202667236
Real Average precision: 0.332838 , Real Average recall: 0.433385 , Average New GO: 3.055263
-----------------------------------------
从AF2DB 20训练+7120微调预测，微调数据为1000个蛋白质
best_epoch:8 AUC:0.9728589238208801 best_Fmax:0.4952494013243441 AUPR:0.4866429596140771 thresh:0.608663022518158
Real Average precision: 0.213495 , Real Average recall: 0.429312 , Average New GO: 5.344499

best_epoch:8 AUC:0.9857218544851157 best_Fmax:0.7165182577764163 AUPR:0.7467417841093973 thresh:0.7595361471176147
Real Average precision: 0.296260 , Real Average recall: 0.786135 , Average New GO: 2.460976

best_epoch:9 AUC:0.9559251438842952 best_Fmax:0.49866350893247485 AUPR:0.5249193227714437 thresh:0.6502326130867004
Real Average precision: 0.244985 , Real Average recall: 0.495170 , Average New GO: 5.274613

PPI:
7120上直接预测：这个结果其实是调大了阈值，在Recall不降太多的情况下，让precision升高一点的，但是没啥用，大部分预测都是趋近1的
2025-11-21 14:58:30,612 global INFO AUC: 0.9786 AUPR: 0.9583 PRC: 0.9255 REC: 0.8944  Fmax: 0.9097 tmp_thresh: 0.6000
2025-11-21 15:00:15,482 global INFO Real Average recall: 0.840337 , Real Average precision: 0.018758 , Average New PPI: 1929.545315
重新测试了一下（上面是之前网页使用的数据，后续更新了, 但不跑全PPI预测了
2026-01-12 21:15:25,806 global INFO test set:
2026-01-12 21:15:25,822 global INFO AUC: 0.9757 AUPR: 0.9482 PRC: 0.8788 REC: 0.9078  Fmax: 0.8930 tmp_thresh: 0.3889

29413转移ppi后7120上预测（最好）
重新测试了一下
2026-01-03 04:33:00,852 global INFO test set:
2026-01-03 04:33:00,867 global INFO AUC: 0.9787 AUPR: 0.9548 PRC: 0.8947 REC: 0.9080  Fmax: 0.9013 tmp_thresh: 0.3720
用0.5的为阈值，上传到网页
2026-01-05 14:18:29,546 global INFO AUC: 0.9802 AUPR: 0.9575 PRC: 0.9108 REC: 0.9007  Fmax: 0.9057 tmp_thresh: 0.5000
2026-01-05 14:20:16,665 global INFO Real Average recall: 0.798739 , Real Average precision: 0.014945 , Average New PPI: 1891.966424

29413上训练+7120微调2轮：
AUC: 0.9412 AUPR: 0.8932 PRC: 0.7880 REC: 0.8852  Fmax: 0.8338 tmp_thresh: 0.2500
Real Average recall: 0.822284 , Real Average precision: 0.009344 , Average New PPI: 2961.675055

7120实验PPI：
2026-01-16 12:38:11,361 global INFO test set:
2026-01-16 12:38:11,371 global INFO AUC: 0.9640 AUPR: 0.9235 PRC: 0.8571 REC: 0.9068  Fmax: 0.8812 tmp_thresh: 0.2615
阈值设为0.3，预测结果上传到网页
2026-01-16 16:23:59,578 global INFO AUC: 0.9677 AUPR: 0.9298 PRC: 0.8711 REC: 0.9107  Fmax: 0.8905 tmp_thresh: 0.3000
2026-01-16 16:27:01,323 global INFO Real Average recall: 0.704123 , Real Average precision: 0.011737 , Average New PPI: 3043.764796