# 对json配置文件的一些基本操作
# 最后修改日期：
# 2021-06-24

from easygui import *
import sys
import json
import os
import datetime

#CONF_DIR = 'D:/工作文档/网络金融部/2021年工作/个人手机银行/报表/每日报表/'
#CONF_DIR = os.path.dirname(os.path.realpath(__file__))
CONF_DIR = os.getcwd()

# 获取提取报表的年、月、日
def get_rpt_date1():
	# 获取T-2的日期
	mdelta = 2
	i = 0
	while True:
		i += 1
		mdelta1 = input("请输入要提取几天前的报表？请输入：")
		if (mdelta1.isdigit()):
			break
		if (i > 3):
			print('三次输错，默认提取T-2日的报表。')
			mdelta1 = 0
			break
		print('请重新输入，必须输入正确的数字。')

	if (int(mdelta1) < 2):
		pass
	else:
		mdelta = int(mdelta1)
	time1 = datetime.datetime.now() + datetime.timedelta(days=-mdelta)

	year = time1.strftime("%Y")
	month = time1.strftime("%m")
	day = time1.strftime("%d")
	return year, month, day


def get_rpt_date():
	# 从配置文件中读取日期
	config = get_rpt_conf()
	date = []
	date = config['date']

	# 默认提取当前日期之前2日的报表
	if (date == ''):
		# 获取T-2的日期
		time1 = datetime.datetime.now() + datetime.timedelta(days=-2)

		year = time1.strftime("%Y")
		month = time1.strftime("%m")
		day = time1.strftime("%d")
	else:
		year = date[0:4]
		month = date[4:6]
		day = date[6:8]

	return year, month, day


# 默认提取当前日期的前二日日期T-2
def get_default_rpt_date():
	time1 = datetime.datetime.now() + datetime.timedelta(days=-2)

	year = time1.strftime("%Y")
	month = time1.strftime("%m")
	day = time1.strftime("%d")

	return year, month, day


# 初始化配置信息，向json配置文件写入配置信息
def init_rpt_conf():
	conf = {
		'date': '20210607'
		, 'target_directory': 'D:/工作文档/网络金融部/2021年工作/个人手机银行/报表/每日报表/'
		, 'download.default_directory': 'D:\\tmp\\report\\'
		, 'name': 'yangzhiwei.nx'
		, 'keyword': 'xxxxxxxxxx'
	}

	conf_dir = CONF_DIR
	conf_filename = os.path.join(conf_dir, 'config.json')
	jsObj = json.dumps(conf)

	with open(conf_filename, 'w') as f:
		f.write(jsObj)


# 从json配置文件获取提取所有配置项
def get_rpt_conf():
	conf_dir = CONF_DIR
	conf_filename = os.path.join(conf_dir, 'config.json')

	fr = open(conf_filename)

	config = json.load(fr)

	return config


# 修改、增加配置项
def write_conf(config):
	conf_dir = CONF_DIR
	conf_filename = os.path.join(conf_dir, 'config.json')
	jsObj = json.dumps(config)

	with open(conf_filename, 'w') as f:
		f.write(jsObj)


# 修改密码
def modify_pass(config):
	i = 0
	# reply = passwordbox("Demo of password box WITH default"
	#					 + "\n\nEnter your secret password",
	#					 "Member Logon", "alfie")
	# print("Reply was: {!s}".format(reply))

	while True:
		i += 1
		reply = passwordbox("\n\n请输入uass密码(至少6位)：\n",
						"Uass登录密码")
		password = '{!s}'.format(reply)
		#password = input("请输入uass密码(至少6位)：")
		if (len(password) > 6):
			break
		if (i > 3):
			print('三次输入不符合要求，你有点累了......')
			break

	password_asc = list(map(ord, password))
	config['keyword'] = password_asc
	return config


# 解密
def decry_pass(old_pass):
	password = "".join(map(chr, old_pass))
	return password


# 从json配置文件获取提取所有配置项
def disp_rpt_conf():
	conf_dir = CONF_DIR
	conf_filename = os.path.join(conf_dir, 'config.json')
	text_snippet=""

	fr = open(conf_filename)

	config = json.load(fr)
	for i in config.keys():
		print('配置项: %s	 当前值: %s' % (i, config[i]))
		#text_snippet=text_snippet+'{!s}'.format(config[i])
		text_snippet=text_snippet+'配置项: {!s}		\n当前值: {!s}'.format(i, config[i])
		text_snippet=text_snippet+'\n\n'


	title = "配置信息："
	msg = " " * 10
	reply = textbox(msg, title, text_snippet)

#修改配置信息
def modify_config():
	config = get_rpt_conf()

	msg = "请输入要修改的配置信息"
	title = "报表制作系统"
	fieldNames = ["登录用户名", "发件人邮箱号", "收件人邮箱", "报表制作目录", "报表归档目录","报表数据临时保存目录", "报表文件名称", "高级网点报表菜单名称", "高级机构报表菜单名称"]
	fieldValues = list()  # we start with blanks for the values
	fieldValues.append(config['name'])
	fieldValues.append(config['mail_by'])
	fieldValues.append(config['recivers'])
	fieldValues.append(config['target_directory'])
	fieldValues.append(config['rpt_lib'])
	fieldValues.append(config['download.default_directory'])
	fieldValues.append(config['rpt_filename'])
	fieldValues.append(config['GJCX_bbm_wd'])
	fieldValues.append(config['GJCX_bbm_jg'])

	fieldValues = multenterbox(msg, title, fieldNames,fieldValues)

	# make sure that none of the fields was left blank
	while True:
		if fieldValues is None:
			break
		errs = list()
		for n, v in zip(fieldNames, fieldValues):
			if v.strip() == "":
				errs.append('"{}" 是一个必须输入的字段。'.format(n))
		if not len(errs):
			break  # no problems found
		fieldValues = multenterbox(
			"\n".join(errs), title, fieldNames, fieldValues)

	print("Reply was: {}".format(fieldValues))
	#return fieldValues
	config['name']=fieldValues[0]
	config['mail_by']=fieldValues[1]
	config['recivers']=fieldValues[2]
	config['target_directory']=fieldValues[3]
	config['rpt_lib']=fieldValues[4]
	config['download.default_directory']=fieldValues[5]
	config['rpt_filename']=fieldValues[6]
	config['GJCX_bbm_wd']=fieldValues[7]
	config['GJCX_bbm_jg']=fieldValues[8]

	#重新写入配置文件
	write_conf(config)

	#显示配置信息
	disp_rpt_conf()

#删除某一个配置项
def del_config(field_name):

	config = get_rpt_conf()

	#删除某一个键值
	del config[field_name]

	#重新写入配置文件
	write_conf(config)

	#显示配置信息
	disp_rpt_conf()

#删除某一个配置项
def add_config(field_name):

	config = get_rpt_conf()

	#删除某一个键值
	config[field_name] = '  '

	#重新写入配置文件
	write_conf(config)

	#显示配置信息
	disp_rpt_conf()

#检查配置信息中相关的目录、文件是否存在
def check_config():

	driver_path =os.getcwd()+r"\chromedriver.exe"
	if not os.path.exists(driver_path) :
		 msgbox("报表数据下载驱动程序：“{!s}” 不存在，请将该程序拷贝到当前目录下。".format(driver_path),"提示信息：")
		 return False

	config = get_rpt_conf()

	first_run=config['First_run']

	if first_run =='0':
		target_directory = config['target_directory']
		rpt_lib = config['rpt_lib']
		down_dir = config['download.default_directory']
		if not os.path.exists(target_directory) :
			 msgbox("您配置的报表制作目录：“{!s}” 不存在，已使用默认目录，您可以在修改配置中进行修改。".format(target_directory),"提示信息：")
			 os.system("mkdir D:\\tmp\\rpt")
			 config['target_directory']="D:\\tmp\\rpt\\"
			 dir_flag = False
		elif not os.path.exists(rpt_lib) :
			 msgbox("您配置的报表归档目录：“{!s}” 不存在，已使用默认目录，您可以在修改配置中进行修改。".format(rpt_lib),"提示信息：")
			 os.system("mkdir D:\\tmp\\rpt_lib")
			 config['rpt_lib']="D:\\tmp\\rpt_lib\\"
			 dir_flag = False
		elif not os.path.exists(down_dir) :
			 msgbox("您配置的报表数据临时保存目录：“{!s}” 不存在，已使用默认目录，您可以在修改配置中进行修改。".format(down_dir),"提示信息：")
			 os.system("mkdir D:\\tmp\\report")
			 config['download.default_directory']="D:\\tmp\\report\\"
			 dir_flag = False
		else:
			dir_flag = True

		if not dir_flag :
			config['First_run'] = '1'
			#重新写入配置文件
			write_conf(config)

			#显示配置信息
			disp_rpt_conf()
		else:
			print('配置项中需要的目录都存在。')

	return True


if __name__ == '__main__':
	# config = get_rpt_conf()
	# modify_pass(config)

	# write_conf(config)
	#disp_rpt_conf()
	modify_config()
	#add_config('GJCX_bbm_wd')
	#add_config('GJCX_bbm_jg')
	#del_config('test')

