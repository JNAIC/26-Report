import time

val_alpha_dic={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
               'A':10,'B':11,'C':12,'D':13,'E':14,'F':15,'G':16,'H':17,'I':18,'J':19,'K':20,'L':21,'M':22,
               'N':23,'O':24,'P':25,'Q':26,'R':27,'S':28,'T':29,'U':30,'V':31,'W':32,'X':33,'Y':34,'Z':35,
               'a':36,'b':37,'c':38,'d':39,'e':40,'f':41,'g':42,'h':43,'i':44,'j':45,'k':46,'l':47,'m':48,
               'n':49,'o':50,'p':51,'q':52,'r':53,'s':54,'t':55,'u':56,'v':57,'w':58,'x':59,'y':60,'z':61}
val_digit_dic={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
               10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',
               23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z',
               36:'a',37:'b',38:'c',39:'d',40:'e',41:'f',42:'g',43:'h',44:'i',45:'j',46:'k',47:'l',48:'m',
               49:'n',50:'o',51:'p',52:'q',53:'r',54:'s',55:'t',56:'u',57:'v',58:'w',59:'x',60:'y',61:'z'}

def check_num_logical(init_number,init_system)->bool:
    #检测数字合法性的核心函数，先查是否为空字符串，再逐个遍历字符串的每一个字符，判断一个字符是否是合法字符，以及该数在当前进制下是否合法
    if len(init_number)==0:
        return False
    for i in init_number:
        if '0'<=i<='9':
            if int(i)>=init_system:
                return False
        elif 'A'<=i<='Z' or 'a'<=i<='z':
            if val_alpha_dic[i]>=init_system:
                return False
        else:
            return False
    return True

def system_input(tip_text):
    system=input(f"{tip_text}").strip()
            
    '''try-except检测输入的是否为非数字字符，如果是则执行except进入新循环并提示用户，
    否则先int再判断输入的进制数是否在支持的范围内'''
    try:
        system=int(system)
        if 2<=system<=62:
            return system
        else:
            print("不支持的进制数，请重新输入。")
            return False
    except ValueError:
        print("请输入合法的进制数！")
        return False

def result(init_number,init_system,goal_system)->str:
    '''先对输入的数字字符串逐个处理，将每个数字字符转化为整型状态，
    再通过公式V=d_(k-1)*b**(k-1)+d_(k-2)*b**(k-2)+...+d_1*b**1+d_0计算其基准值（为了好理解其实把基准值当十进制状态的数就行）'''
    lst=[]
    value=0
    for i in init_number:
        d=val_alpha_dic[i]
        value=value*init_system+d

    '''第二步采用除基取余法对获得的基准值进行转化，使其成为对应进制的数字形态，使用列表和join方法增加处理效率'''
    if value==0:
        return '0'
    while value!=0:
        value,r=divmod(value,goal_system)
        lst.append(val_digit_dic[r])
    lst.reverse()
    goal_number="".join(lst)
    return goal_number

def main():
    #输入初始进制数并对输入的进制数合法性进行检验
    while True:
        init_system=system_input("请输入初始数字进制数（目前支持2~62进制）:")
        if init_system==False:
            continue
        else:
            break
    #输入待转化的数字并对输入数字在当前进制下的合法性进行检验
    while True:
        init_number=input("请输入待转化数（输入的数字要在当前进制下合法）：").strip()
        if check_num_logical(init_number,init_system):
            break
        else:
            print("该数字不合法！")
            continue

    #输入要转化的目标进制数并对输入的进制数合法性进行检验
    while True:
        goal_system=system_input("请输入要转化的目标进制（目前支持2~62进制）:")
        if goal_system==False:
            continue
        else:
            break

    #将数字转换为指定进制形式并输出
    print(f"初始数字：{init_number}",f"({init_system}进制)",sep="")
    start_time=time.perf_counter()
    goal_number=result(init_number,init_system,goal_system)
    end_time=time.perf_counter()
    print(f"目标数字：{goal_number}",f"({goal_system}进制)",sep="")
    print(f"转化耗时：{end_time-start_time:.6g}s")

if __name__=='__main__':
    main()