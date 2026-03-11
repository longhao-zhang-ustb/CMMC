def get_single_evaluate(precision, recall, f1score, data_type):
    if data_type == 'semeval_9class':
        return {
            'Message-Topic': {'precision': round(precision[0],4), 'recall': round(recall[0],4), 'f1-score': round(f1score[0],4)}, 
            'Product-Producer': {'precision': round(precision[1],4), 'recall': round(recall[1],4), 'f1-score': round(f1score[1],4)},
            'Instrument-Agency': {'precision': round(precision[2],4), 'recall': round(recall[2],4), 'f1-score': round(f1score[2],4)}, 
            'Entity-Destination': {'precision': round(precision[3],4), 'recall': round(recall[3],4), 'f1-score': round(f1score[3],4)},  
            'Cause-Effect': {'precision': round(precision[4],4), 'recall': round(recall[4],4), 'f1-score': round(f1score[4],4)}, 
            'Component-Whole': {'precision': round(precision[5],4), 'recall': round(recall[5],4), 'f1-score': round(f1score[5],4)},
            'Entity-Origin': {'precision': round(precision[6],4), 'recall': round(recall[6],4), 'f1-score': round(f1score[6],4)}, 
            'Member-Collection': {'precision': round(precision[7],4), 'recall': round(recall[7],4), 'f1-score': round(f1score[7],4)},
            'Content-Container': {'precision': round(precision[8],4), 'recall': round(recall[8],4), 'f1-score': round(f1score[8],4)}
        }
    elif data_type == 'semeval_18class':
        return {
            'Message-Topic(e1, e2)': {'precision': round(precision[0],4), 'recall': round(recall[0],4), 'f1-score': round(f1score[0],4)}, 
            'Message-Topic(e2, e1)': {'precision': round(precision[1],4), 'recall': round(recall[1],4), 'f1-score': round(f1score[1],4)},
            'Product-Producer(e1, e2)': {'precision': round(precision[2],4), 'recall': round(recall[2],4), 'f1-score': round(f1score[2],4)}, 
            'Product-Producer(e2, e1)': {'precision': round(precision[3],4), 'recall': round(recall[3],4), 'f1-score': round(f1score[3],4)},
            'Instrument-Agency(e1, e2)': {'precision': round(precision[4],4), 'recall': round(recall[4],4), 'f1-score': round(f1score[4],4)}, 
            'Instrument-Agency(e2, e1)': {'precision': round(precision[5],4), 'recall': round(recall[5],4), 'f1-score': round(f1score[5],4)},
            'Entity-Destination(e1, e2)': {'precision': round(precision[6],4), 'recall': round(recall[6],4), 'f1-score': round(f1score[6],4)}, 
            'Entity-Destination(e2, e1)': {'precision': round(precision[7],4), 'recall': round(recall[7],4), 'f1-score': round(f1score[7],4)},
            'Cause-Effect(e1, e2)': {'precision': round(precision[8],4), 'recall': round(recall[8],4), 'f1-score': round(f1score[8],4)},
            'Cause-Effect(e2, e1)': {'precision': round(precision[9],4), 'recall': round(recall[9],4), 'f1-score': round(f1score[9],4)},
            'Component-Whole(e1, e2)': {'precision': round(precision[10],4), 'recall': round(recall[10],4), 'f1-score': round(f1score[10],4)},
            'Component-Whole(e2, e1)': {'precision': round(precision[11],4), 'recall': round(recall[11],4), 'f1-score': round(f1score[11],4)}, 
            'Entity-Origin(e1, e2)': {'precision': round(precision[12],4), 'recall': round(recall[12],4), 'f1-score': round(f1score[12],4)},
            'Entity-Origin(e2, e1)': {'precision': round(precision[13],4), 'recall': round(recall[13],4), 'f1-score': round(f1score[13],4)}, 
            'Member-Collection(e1, e2)': {'precision': round(precision[14],4), 'recall': round(recall[14],4), 'f1-score': round(f1score[14],4)},
            'Member-Collection(e2, e1)': {'precision': round(precision[15],4), 'recall': round(recall[15],4), 'f1-score': round(f1score[15],4)}, 
            'Content-Container(e1, e2)': {'precision': round(precision[16],4), 'recall': round(recall[16],4), 'f1-score': round(f1score[16],4)},
            'Content-Container(e2, e1)': {'precision': round(precision[17],4), 'recall': round(recall[17],4), 'f1-score': round(f1score[17],4)}
        }
    elif data_type == 're_traced':
        return {
            "org2miscmulti": {'precision': round(precision[0],4), 'recall': round(recall[0],4), 'f1-score': round(f1score[0],4)},
            "org2locmulti": {'precision': round(precision[1],4), 'recall': round(recall[1],4), 'f1-score': round(f1score[1],4)},
            "org2org": {'precision': round(precision[2],4), 'recall': round(recall[2],4), 'f1-score': round(f1score[2],4)},
            "org2per": {'precision': round(precision[3],4), 'recall': round(recall[3],4), 'f1-score': round(f1score[3],4)},
            "per2miscmulti": {'precision': round(precision[4],4), 'recall': round(recall[4],4), 'f1-score': round(f1score[4],4)},
            "per2locmulti": {'precision': round(precision[5],4), 'recall': round(recall[5],4), 'f1-score': round(f1score[5],4)},
            "per2org": {'precision': round(precision[6],4), 'recall': round(recall[6],4), 'f1-score': round(f1score[6],4)},
            "per2per": {'precision': round(precision[7],4), 'recall': round(recall[7],4), 'f1-score': round(f1score[7],4)}
        }
    elif data_type == 'i2b2':
        return {
            "reason-drug": {'precision': round(precision[0],2), 'recall': round(recall[0],2), 'f1-score': round(f1score[0],2)},
            "route-drug": {'precision': round(precision[1],2), 'recall': round(recall[1],2), 'f1-score': round(f1score[1],2)},
            "strength-drug": {'precision': round(precision[2],2), 'recall': round(recall[2],2), 'f1-score': round(f1score[2],2)},
            "frequency-drug": {'precision': round(precision[3],2), 'recall': round(recall[3],2), 'f1-score': round(f1score[3],2)},
            "duration-drug": {'precision': round(precision[4],2), 'recall': round(recall[4],2), 'f1-score': round(f1score[4],2)},
            "form-drug": {'precision': round(precision[5],2), 'recall': round(recall[5],2), 'f1-score': round(f1score[5],2)},
            "dosage-drug": {'precision': round(precision[6],2), 'recall': round(recall[6],2), 'f1-score': round(f1score[6],2)},
            "ade-drug": {'precision': round(precision[7],2), 'recall': round(recall[7],2), 'f1-score': round(f1score[7],2)}
        }
    elif data_type == 'baike':
        return {
            "kinship": {'precision': round(precision[0],2), 'recall': round(recall[0],2), 'f1-score': round(f1score[0],2)},
            "couple": {'precision': round(precision[1],2), 'recall': round(recall[1],2), 'f1-score': round(f1score[1],2)},
            "friends/colleague": {'precision': round(precision[2],2), 'recall': round(recall[2],2), 'f1-score': round(f1score[2],2)},
            "classmate/schoolmate": {'precision': round(precision[3],2), 'recall': round(recall[3],2), 'f1-score': round(f1score[3],2)},
            "antagonism/competition": {'precision': round(precision[4],2), 'recall': round(recall[4],2), 'f1-score': round(f1score[4],2)},
            "student-teacher": {'precision': round(precision[5],2), 'recall': round(recall[5],2), 'f1-score': round(f1score[5],2)}
        }
    elif data_type == 'baidu':
        return {
            "民族": {'precision': round(precision[0],2), 'recall': round(recall[0],2), 'f1-score': round(f1score[0],2)},
            "出生日期": {'precision': round(precision[1],2), 'recall': round(recall[1],2), 'f1-score': round(f1score[1],2)},
            "出生地": {'precision': round(precision[2],2), 'recall': round(recall[2],2), 'f1-score': round(f1score[2],2)},
            "作曲": {'precision': round(precision[3],2), 'recall': round(recall[3],2), 'f1-score': round(f1score[3],2)},
            "所属专辑": {'precision': round(precision[4],2), 'recall': round(recall[4],2), 'f1-score': round(f1score[4],2)},
            "歌手": {'precision': round(precision[5],2), 'recall': round(recall[5],2), 'f1-score': round(f1score[5],2)},
            "作词": {'precision': round(precision[6],2), 'recall': round(recall[6],2), 'f1-score': round(f1score[6],2)},
            "成立日期": {'precision': round(precision[7],2), 'recall': round(recall[7],2), 'f1-score': round(f1score[7],2)},
            "作者": {'precision': round(precision[8],2), 'recall': round(recall[8],2), 'f1-score': round(f1score[8],2)},
            "连载网站": {'precision': round(precision[9],2), 'recall': round(recall[9],2), 'f1-score': round(f1score[9],2)},
            "毕业院校": {'precision': round(precision[10],2), 'recall': round(recall[10],2), 'f1-score': round(f1score[10],2)},
            "出品公司": {'precision': round(precision[11],2), 'recall': round(recall[11],2), 'f1-score': round(f1score[11],2)},
            "主演": {'precision': round(precision[12],2), 'recall': round(recall[12],2), 'f1-score': round(f1score[12],2)},
            "出版社": {'precision': round(precision[13],2), 'recall': round(recall[13],2), 'f1-score': round(f1score[13],2)},
            "国籍": {'precision': round(precision[14],2), 'recall': round(recall[14],2), 'f1-score': round(f1score[14],2)},
            "导演": {'precision': round(precision[15],2), 'recall': round(recall[15],2), 'f1-score': round(f1score[15],2)},
            "上映时间": {'precision': round(precision[16],2), 'recall': round(recall[16],2), 'f1-score': round(f1score[16],2)}
        }
    elif data_type == 'medical':
        return {
            "发病年龄": {'precision': round(precision[0],2), 'recall': round(recall[0],2), 'f1-score': round(f1score[0],2)},
            "发病部位": {'precision': round(precision[1],2), 'recall': round(recall[1],2), 'f1-score': round(f1score[1],2)},
            "遗传因素": {'precision': round(precision[2],2), 'recall': round(recall[2],2), 'f1-score': round(f1score[2],2)},
            "多发地区": {'precision': round(precision[3],2), 'recall': round(recall[3],2), 'f1-score': round(f1score[3],2)},
            "多发季节": {'precision': round(precision[4],2), 'recall': round(recall[4],2), 'f1-score': round(f1score[4],2)},
            "高危因素": {'precision': round(precision[5],2), 'recall': round(recall[5],2), 'f1-score': round(f1score[5],2)},
            "筛查": {'precision': round(precision[6],2), 'recall': round(recall[6],2), 'f1-score': round(f1score[6],2)},
            "临床表现": {'precision': round(precision[7],2), 'recall': round(recall[7],2), 'f1-score': round(f1score[7],2)},
            "传播途径": {'precision': round(precision[8],2), 'recall': round(recall[8],2), 'f1-score': round(f1score[8],2)},
            "病理分型": {'precision': round(precision[9],2), 'recall': round(recall[9],2), 'f1-score': round(f1score[9],2)},
            "发病性别倾向": {'precision': round(precision[10],2), 'recall': round(recall[10],2), 'f1-score': round(f1score[10],2)},
            "药物治疗": {'precision': round(precision[11],2), 'recall': round(recall[11],2), 'f1-score': round(f1score[11],2)},
            "放射治疗": {'precision': round(precision[12],2), 'recall': round(recall[12],2), 'f1-score': round(f1score[12],2)},
            "发病机制": {'precision': round(precision[13],2), 'recall': round(recall[13],2), 'f1-score': round(f1score[13],2)},
            "多发群体": {'precision': round(precision[14],2), 'recall': round(recall[14],2), 'f1-score': round(f1score[14],2)},
            "就诊科室": {'precision': round(precision[15],2), 'recall': round(recall[15],2), 'f1-score': round(f1score[15],2)},
            "辅助检查": {'precision': round(precision[16],2), 'recall': round(recall[16],2), 'f1-score': round(f1score[16],2)},
            "组织学检查": {'precision': round(precision[17],2), 'recall': round(recall[17],2), 'f1-score': round(f1score[17],2)},
            "相关（转化）": {'precision': round(precision[18],2), 'recall': round(recall[18],2), 'f1-score': round(f1score[18],2)},
            "阶段": {'precision': round(precision[19],2), 'recall': round(recall[19],2), 'f1-score': round(f1score[19],2)},
            "治疗后症状": {'precision': round(precision[20],2), 'recall': round(recall[20],2), 'f1-score': round(f1score[20],2)},
            "侵及周围组织转移的症状": {'precision': round(precision[21],2), 'recall': round(recall[21],2), 'f1-score': round(f1score[21],2)},
            "鉴别诊断": {'precision': round(precision[22],2), 'recall': round(recall[22],2), 'f1-score': round(f1score[22],2)},
            "化疗": {'precision': round(precision[23],2), 'recall': round(recall[23],2), 'f1-score': round(f1score[23],2)},
            "预防": {'precision': round(precision[24],2), 'recall': round(recall[24],2), 'f1-score': round(f1score[24],2)},
            "同义词": {'precision': round(precision[25],2), 'recall': round(recall[25],2), 'f1-score': round(f1score[25],2)},
            "相关（症状）": {'precision': round(precision[26],2), 'recall': round(recall[26],2), 'f1-score': round(f1score[26],2)},
            "转移部位": {'precision': round(precision[27],2), 'recall': round(recall[27],2), 'f1-score': round(f1score[27],2)},
            "死亡率": {'precision': round(precision[28],2), 'recall': round(recall[28],2), 'f1-score': round(f1score[28],2)},
            "病理生理": {'precision': round(precision[29],2), 'recall': round(recall[29],2), 'f1-score': round(f1score[29],2)},
            "实验室检查": {'precision': round(precision[30],2), 'recall': round(recall[30],2), 'f1-score': round(f1score[30],2)},
            "病史": {'precision': round(precision[31],2), 'recall': round(recall[31],2), 'f1-score': round(f1score[31],2)},
            "预后状况": {'precision': round(precision[32],2), 'recall': round(recall[32],2), 'f1-score': round(f1score[32],2)},
            "手术治疗": {'precision': round(precision[33],2), 'recall': round(recall[33],2), 'f1-score': round(f1score[33],2)},
            "并发症": {'precision': round(precision[34],2), 'recall': round(recall[34],2), 'f1-score': round(f1score[34],2)},
            "内窥镜检查": {'precision': round(precision[35],2), 'recall': round(recall[35],2), 'f1-score': round(f1score[35],2)},
            "外侵部位": {'precision': round(precision[36],2), 'recall': round(recall[36],2), 'f1-score': round(f1score[36],2)},
            "相关（导致）": {'precision': round(precision[37],2), 'recall': round(recall[37],2), 'f1-score': round(f1score[37],2)},
            "病因": {'precision': round(precision[38],2), 'recall': round(recall[38],2), 'f1-score': round(f1score[38],2)},
            "影像学检查": {'precision': round(precision[39],2), 'recall': round(recall[39],2), 'f1-score': round(f1score[39],2)},
            "预后生存率": {'precision': round(precision[40],2), 'recall': round(recall[40],2), 'f1-score': round(f1score[40],2)},
            "风险评估因素": {'precision': round(precision[41],2), 'recall': round(recall[41],2), 'f1-score': round(f1score[41],2)},
            "辅助治疗": {'precision': round(precision[42],2), 'recall': round(recall[42],2), 'f1-score': round(f1score[42],2)},
            "发病率": {'precision': round(precision[43],2), 'recall': round(recall[43],2), 'f1-score': round(f1score[43],2)}
        }
