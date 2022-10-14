### Python + Steamlit 快速开发可视化 web 页面
##在命令窗口输入streamlit run test.py。
##浏览器输入：http://localhost:8501/

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
#from datetime import time
import time

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#import time
import os
#import seaborn as sns
import plotly.figure_factory as ff
import altair as alt
from PIL import Image
import base64

import urllib.request

#import example_scenes
import rpt_config as cfg  # 引入本地的操作配置文件的程序包
import test_pic as tpc # 引入本地子模块程序包
import altair_app as alt_app

# 设置网页标题，以及使用宽屏模式（必须在每一个app的最前面设置）
st.set_page_config(
    page_title="手机银行报表查询、制作平台",
    layout="wide"
)

def test_base():  #基本功能展示
    # 展示文本；文本直接使用Markdown语法
    st.markdown("""
                - 这是
                - 一个
                - 无序列表
                """)

    # 展示pandas数据框
    st.dataframe(pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"]))

    # 展示matplotlib绘图
    arr = np.random.normal(1, 1, size=100)
    fig, ax = plt.subplots()
    plt.hist(arr, bins=20)
    plt.title("matplotlib plot")
    st.pyplot(fig)

    # 加入交互控件，如输入框
    number = st.number_input("Insert a number", 123, key = 'first_number')
    st.write("输入的数字是：", number)

    #文字、公式
    st.title("st.title")
    st.header("st.header")
    st.subheader("st.subheader")

    st.markdown("# markdown一级标题")
    st.markdown("## markdown二级标题")
    st.markdown("### markdown三级标题")

    # ------------------------------------

    st.write("st.write")
    st.markdown("markdown普通文字")
    st.text("st.text")

    # ------------------------------------

    st.markdown("""
    Markdown列表：
    - hello
    - world
    - China
    """)

    st.markdown("***markdown粗斜体***")

    # ------------------------------------

    st.latex("\sum_{i=1}^{n}")
    st.latex(r''' e^{i\pi} + 1 = 0 ''')

    # ------------------------------------

    st.write(1234)
    st.write("1234")
    st.write("1 + 1 = ", 2)

    # --------------------------------
    st.json({'foo':'bar','fu':'ba'})

    numstr = '程序开始时输入的数字是：  '+ str(st.session_state.first_number)
    st.subheader(numstr)

def test_write(): #write功能展示
    '''
    st.write()
    st.write()是一个泛型函数，根据传入对象不同采取不同的展示方式，比如传入pandas.DataFrame时，st.write(df)默认调用st.dataframe()，传入markdown时，st.write(markdown)默认调用st.markdown()；跟R的泛型函数非常类似。可传入的对象有:

    write(data_frame) : Displays the DataFrame as a table.
    write(func) : Displays information about a function.
    write(module) : Displays information about the module.
    write(dict) : Displays dict in an interactive widget.
    write(obj) : The default is to print str(obj).
    write(mpl_fig) : Displays a Matplotlib figure.
    write(altair) : Displays an Altair chart.
    write(keras) : Displays a Keras model.
    write(graphviz) : Displays a Graphviz graph.
    write(plotly_fig) : Displays a Plotly figure.
    write(bokeh_fig) : Displays a Bokeh figure.
    write(sympy_expr) : Prints SymPy expression using LaTeX.
    write(markdown):
    '''

    # 字典
    st.write({"a": [1, 2, 3],
              "b": [2, 3, 4]})

    # pandas数据框
    st.write(pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [4, 5, 6, 7, 8]
    }))

    # Markdown文字
    st.write("Hello, *World!* :sunglasses:")

    # 绘图对象
    df = pd.DataFrame(
        np.random.randn(200, 3),
        columns=["a", "b", "c"]
    )

    c = alt.Chart(df).mark_circle().encode(
        x="a", y="b", size="c", color="c", tooltip=["a", "b", "c"])
    st.write(c)

def test_table(): #动态表格展示
    # 动态表格（表格太大时只展示一部分，可滑动表格下方滑动条查看不同部分）
    # st.write默认调用st.dataframe()
    df = pd.DataFrame(
        np.random.randn(10, 20),
        columns=('col %d' % i for i in range(20)))

    st.write(df)

    # 静态表格（展示表格全部内容，太大时滑动App界面下方的滑动条查看不同部分）
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [4, 5, 6, 7, 8]
    })

    st.table(df)

    #pandas.DataFrame的style也可正常显示
    df = pd.DataFrame(
        np.random.randn(10, 20),
        columns=('col %d' % i for i in range(20)))

    st.dataframe(df.style.highlight_max(axis=0))

    #Code
    #仅展示Code，Code不执行
    code = """
    def sum_(x):
        return np.sum(x)
    """
    st.code(code, language="python")

    code = """
    for (i i 1:10) {
        print(i)
    }
    """
    st.code(code, language="r")

    st.markdown("""
    ​```python
    print("hello")
    """)
    #展示Code，同时执行Code；需要将code放入st.echo()内
    with st.echo():
        for i in range(5):
            st.write("hello")

    #使用缓存，第一次加载后，下次调用，直接从缓存中提取数据
    @st.cache
    def load_metadata():
        #DATA_URL = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/labels.csv.gz"
        DATA_URL = "labels.csv.gz"
        return pd.read_csv(DATA_URL, nrows=1000)

    @st.cache
    def create_summary(metadata, summary_type):
        one_hot_encoded = pd.get_dummies(metadata[["frame", "label"]],
                            columns=["label"])
        return getattr(one_hot_encoded.groupby(["frame"]), summary_type)()

    # Piping one st.cache function into another forms a computation DAG.
    summary_type = st.selectbox("Type of summary:", ["sum", "any"])
    metadata = load_metadata()
    #metadata.to_excel('labels.xlsx',index=False)
    summary = create_summary(metadata, summary_type)
    st.write('## Metadata', metadata, '## Summary', summary)

    #远程提取数据，并缓存，使用select来选择要显示的数据内容。
    # Reuse this data across runs!
    read_and_cache_csv = st.cache(pd.read_csv)

    #BUCKET = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"
    BUCKET = ""
    data = read_and_cache_csv(BUCKET + "labels.csv.gz", nrows=1000)
    desired_label = st.selectbox('Filter to:', ['car', 'truck'])
    st.write(data[data.label == desired_label])

    #从S3下载数据，并根据选择，展示数据
    DATE_COLUMN = 'date/time'
    # DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
    #             'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

    DATA_URL = 'uber-raw-data-sep14.csv'

    @st.cache
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        return data

    data_load_state = st.text('Loading data...')
    data = load_data(1000)
    #data.to_csv('uber-raw-data-sep14.csv',index=False)

    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)

    st.subheader('Number of pickups by hour')
    hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
    st.bar_chart(hist_values)

    #展示数据
    option = st.selectbox(
        'Which number do you like best?',
          [1,2,3,4,5])
    # Some number in the range 0-23
    hour_to_filter = st.slider('hour', 0, 23, 17)
    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

    st.subheader('Map of all pickups at %s:00' % hour_to_filter)
    st.map(filtered_data)



    dataframe = pd.DataFrame(
        np.random.randn(10, 20),
        columns=('col %d' % i for i in range(20)))

    '''
    dataframe 显示方式一：sr.write
    '''
    st.write(dataframe)

    '''
    dataframe 显示方式二：直接键入最终结果dataframe
    '''
    dataframe

    '''
    dataframe 显示方式三：st.dataframe
    '''
    st.dataframe(dataframe.style.highlight_max(axis=0))

    '''
    dataframe 显示方式四：st.table
    最丑的一种方式，会变成无页面约束的宽表
    '''
    dataframe = pd.DataFrame(
        np.random.randn(10, 20),
        columns=('col %d' % i for i in range(20)))
    st.table(dataframe)

def test_control(): #控件功能展示
    #各种控件

    # 数字输入框
    number = st.number_input("Insert a number", 123)

    # 单行文本输入框
    word = st.text_input("Insert a word", "123")
    st.write("The number is", number, "The word is", word)

    # 多行文本输入框
    st.text_area("Text to analyze", "I love China")

    # 日期输入框
    st.date_input("Insert a date")

    # 时间输入框
    st.time_input("Insert a time")

    # 向表单插入元素
    with st.form("my_form1"):
        st.write("我在 1 框框里~")
        slider_val = st.slider("框框滑块")
        checkbox_val = st.checkbox("pick me")
        # Every form must have a submit button.
        submitted = st.form_submit_button("1-Submit")

    # 乱序插入元素
    form = st.form("my_form2")
    form.slider("我在 2 框框里~")
    st.slider("我在外面")
    # Now add a submit button to the form:
    form.form_submit_button("2-Submit")


    # 点击按钮
    number = st.button("click it")
    st.write("返回值:", number)

    # 滑动条
    x = st.slider("Square", min_value=0, max_value=80)
    st.write(x, "squared is", np.power(x, 2))

    #或
    x = st.slider('x')
    st.write(x, 'squared is', x * x)

    """ ### 4.6 拉选框


    包括：
    - 常规滑块 - range slider
    - 时间滑块 - time slider
    - 日期选项 - datetime slider

    """


    age = st.slider('How old are you?', 0, 130, 25)
    st.write("I'm ", age, 'years old')

    # 常规滑块 - range slider
    values = st.slider(
        'Select a range of values',
        0.0, 100.0, (25.0, 75.0))
    st.write('Values:', values)


    # 时间滑块 - time slider
    from datetime import time  #此处的time功能是datetime中time模块特有
    appointment = st.slider(
         "Schedule your appointment:",
         value=(time(11, 30), time(12, 45)))
    st.write("You're scheduled for:", appointment)

    # 日期选项 - datetime slider
    start_time = st.slider(
         "请选择开始日期：",
         value=datetime.datetime(2019, 1, 1),
         format="YYYY-MM-DD")
    st.write("开始日期:", start_time)

    # 日期选项 - datetime slider
    start_time = st.slider(
         "请选择结束日期：",
         value=datetime.datetime(2021, 12, 1),
         format="YYYY-MM-DD")
    st.write("结束日期:", start_time)


    # 常规
    color = st.select_slider(
         'Select a color of the rainbow',
         options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
    st.write('My favorite color is', color)

    # range select slider 区域范围的选择滑块
    start_color, end_color = st.select_slider(
        'Select a range of color wavelength',
         options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
         value=('red', 'blue'))
    st.write('You selected wavelengths between', start_color, 'and', end_color)


    # 单文件载入
    uploaded_file = st.file_uploader("单文件载入上传")
    if uploaded_file is not None:
         # To read file as bytes:
         bytes_data = uploaded_file.read()
         st.write(bytes_data)

         # To convert to a string based IO:
         stringio = StringIO(uploaded_file.decode("utf-8"))
         st.write(stringio)

         # To read file as string:
         string_data = stringio.read()
         st.write(string_data)

         # Can be used wherever a "file-like" object is accepted:
         st.write(uploaded_file)
         dataframe = pd.read_csv(uploaded_file)
         st.write(dataframe)

    # 多文件载入
    uploaded_files = st.file_uploader("多个文件载入上传", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        st.write(bytes_data)


    #颜色选择
    color = st.color_picker('颜色选择：', '#00f900')
    st.write('The current color is', color)


    '''
    ## 5 控制组件 - Control flow

    '''


    """ ### 5.1 输入框
        只有输入了，才会继续进行下去...
        """

    name = st.text_input('Name')
    if not name:
        st.warning('Please input a name.')
        st.stop()
    st.success(f'Thank you for inputting a name. {name}')

    #多个选择框 - 选上了就会上记录
    options = st.multiselect(
        'What are your favorite colors',
        ['Green', 'Yellow', 'Red', 'Blue'],
        ['Yellow', 'Red'])
    st.write('You selected:', options)

    # 检查框
    res = st.checkbox("I agree")
    st.write(res)

    # 单选框
    st.selectbox("Which would you like", [1, 2, 3])

    # 单选按钮
    st.radio("Which would you like", [1, 2, 3])

    # 多选框
    selector = st.multiselect("Which would you like", [1, 2, 3])
    st.write(selector)

    ## 气球效果
    #st.balloons()

    #上传csv文件
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)

    #下载：目前Streamlit还没有专门的下载控件，下载pandas Dataframe为csv文件可通过以下方式实现
    #点击Download CSV File便可下载文件
    data = [(1, 2, 3)]
    df = pd.DataFrame(data, columns=["Col1", "Col2", "Col3"])
    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">下载 CSV 文件</a> (点击鼠标右键，另存为 &lt;XXX&gt;.csv)'
    st.markdown(href, unsafe_allow_html=True)

    #侧边栏控件
    #以上控件大部分都有对应的侧边栏形式，如上述st.selectbox若想放置在侧边栏，可使用st.sidebar.selectbox
    # 单选框
    selector1=st.sidebar.selectbox("请选择你喜欢的选项：", [1, 2, 3], key="1")

    # 单选按钮
    selector2=st.sidebar.radio("请选择你喜欢的选项：", [1, 2, 3], key="1")

    # 多选框
    selector = st.sidebar.multiselect("请选择你喜欢的选项：", [1, 2, 3], key="3")
    st.write("你的选项分别是：\n",selector1,selector2,selector)

def test_plot(): #绘图、图片功能展示
    '''
    绘图、图片、音频、视频
    支持的绘图库：

    streamlit自带绘图：st.line_chart()、st.bar_chart()、st.area_chart(data)等
    matplotlib或seaborn：st.pyplot()
    altair：st.altair_chart()
    vega: st.vega_lite_chart()
    plotly: st.plotly_chart()
    bokeh: st.bokeh_chart()

    st.pydeck_chart(data)
    st.deck_gl_chart(data)
    st.graphviz_chart(data)
    st.map(data)

    '''
    st.line_chart(np.random.randn(10, 2))

    chart_data = pd.DataFrame(
        np.random.randn(50, 3),
        columns=["a", "b", "c"]
    )
    st.bar_chart(chart_data)

    #matplotlib或seaborn绘图
    st.markdown("# matplotlib绘图")

    arr = np.random.normal(1, 1, size=100)
    fig, ax = plt.subplots()
    #ax.hist(arr, bins=20)
    plt.hist(arr, bins=20)  #此语句相当于上面两句的功能。
    st.pyplot(fig)

    # st.markdown("# seaborn绘图")

    # tips = sns.load_dataset("tips")
    # sns.set(style="darkgrid")
    # sns.scatterplot(x="total_bill", y="tip", hue="smoker", data=tips)
    # st.pyplot()

    #plotly绘图
    x1 = np.random.randn(200) - 2
    x2 = np.random.randn(200)
    x3 = np.random.randn(200) + 2

    hist_data = [x1, x2, x3]
    group_labels = ["Group 1", "Group 2", "Group 3"]
    # fig = ff.create_distplot(
    #     hist_data, group_labels,
    #     bin_size=[0.1, 0.25, 0.5])

    # st.markdown("# plotly绘图")
    # st.plotly_chart(fig)

    #地图
    # 绘制1000个点的坐标
    map_data = pd.DataFrame(
        #np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        #np.random.randn(1000, 2) / [50, 50] + [32.37, 118.4],
        np.random.randn(1000, 2) / [50, 50] + [38.48, 106.29],
        columns=['lat', 'lon'])

    st.map(map_data)

    #状态
    st.error("错误显示为")
    st.warning("警告显示为")
    st.info("普通消息显示为")
    st.success("成功消息显示为")

    #展示进度
    #占位符
    #占位符可在后续向其中添加内容
    # 添加两个占位符
    slot1 = st.empty()
    slot2 = st.empty()

    # 占位符中插入文字
    time.sleep(0.5)
    slot1.markdown("# This will appear")

    # 占位符中画图
    time.sleep(0.5)
    slot2.line_chart(np.random.randn(20, 2))

    #进度条
    # 添加占位符
    placeholder = st.empty()
    # 创建进度条
    bar = st.progress(0)

    for i in range(100):
        time.sleep(0.05)
        # 不断更新占位符的内容
        placeholder.text(f"Iteration {i+1}")
        # 不断更新进度条
        bar.progress(i + 1)

    # 状态
    st.success("完成。")

    #等待条
    with st.spinner("请稍侯..."):
        for i in range(100):
            print("hello")
            time.sleep(0.05)

    st.success("完成!")

    #动态扩增表格
    df1 = pd.DataFrame(
        np.random.randn(5, 5),
        columns=("col %d" % i for i in range(5))
    )
    tb_table = st.table(df1)

    for i in range(10):
        df2 = pd.DataFrame(
            np.random.randn(1, 5),
            columns=("col %d" % i for i in range(5))
        )
        tb_table.add_rows(df2)
        time.sleep(0.5)

    #动态折线图
    pb = st.progress(0)
    status_txt = st.empty()
    chart = st.line_chart(np.random.randn(10, 2))

    for i in range(100):
        pb.progress(i)
        new_rows = np.random.randn(10, 2)
        status_txt.text(
            "The latest number is: %s" % new_rows[-1, 1]
        )
        chart.add_rows(new_rows)
        time.sleep(0.05)

    #缓存
    @st.cache() # 使用缓存
    def compute_long_time():
        SUM = 0
        for i in range(100):
            SUM += i
            time.sleep(0.05)
        return SUM
    #第一次调用函数，费时较长
    st.write(compute_long_time())

    #再次调用该函数，瞬间出结果
    st.write(compute_long_time())

def test_audio_video(): #音频、视频功能展示

    #Magic commands
    #Streamlit提供了魔法方法，对于某些内容，直接书写便会自动调用st.write()
    "***hello world1***"

    """
    # This is the document title1

    This is some _markdown_.1
    """

    # ---------------------------------

    st.write("***hello world2***")

    st.write("""
    # This is the document title2

    This is some _markdown_.2
    """)

    # ---------------------------------

    st.markdown("***hello world3***")

    st.markdown("""
    # This is the document title3

    This is some _markdown_.3
    """)


    #展示df和x，两者效果相同
    df = pd.DataFrame({"col1": [1, 2, 3]})
    df

    x = 10
    "x", x

    # ---------------------------------

    df = pd.DataFrame({"col1": [1, 2, 3]})
    st.write(df) # 默认调用st.dataframe()

    x = 10
    st.write("x", x)

    #播放音频
    audio_file = open('01.马斯奈泰伊思螟想曲.mp3', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')

    #播放视频
    video_file = open('VID_20210920_194519.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

def test_explor(): #网页布局
    # with写法
    with st.container():
       st.write("This is inside the container")
       # You can call any Streamlit command, including custom components:
       st.bar_chart(np.random.randn(50, 3))

    st.write("This is outside the container")

    # 常规写法
    container = st.container()
    container.write("This is inside the container")
    st.write("This is outside the container")

    # Now insert some more in the container
    container.write("This is inside too")

    '''
    分列展示
    streamlit.beta_columns(spec)

    以并排列的形式插入容器。插入多个并排放置的多元素容器，并返回容器对象列表。

    要向返回的容器添加元素，可以使用“with”表示法(首选)，或者直接调用返回对象的方法。
    '''

    col1, col2, col3 = st.columns(3)

    with col1:
       st.header("A cat")
       st.image("https://static.streamlit.io/examples/cat.jpg", use_column_width=True)

    with col2:
       st.header("A dog")
       st.image("https://static.streamlit.io/examples/dog.jpg", use_column_width=True)

    with col3:
       st.header("An owl")
       st.image("https://static.streamlit.io/examples/owl.jpg", use_column_width=True)

    #按照比例分列展示
    col1, col2 = st.columns([3, 1])
    data = np.random.randn(10, 1)

    col1.subheader("A wide column with a chart")
    col1.line_chart(data)

    col2.subheader("A narrow column with the data")
    col2.write(data)

    #折叠/展开

    st.line_chart({"data": [1, 5, 2, 6, 2, 1]})

    with st.expander("See explanation"):
        st.write("""
            The chart above shows some numbers I picked for you.
            I rolled actual dice for these, so they're *guaranteed* to
            be random.
        """)
        st.image("https://static.streamlit.io/examples/dice.jpg")

def test_pic(): #图像目标检测demo网页

    #直接读入markdown文件
    #之前st.markdown可以直接写，但是如果很长的markdown比较麻烦，于是乎，最好可以先保存一个instructions_yolov3.md文件，然后读入。
    @st.cache(show_spinner=False)
    def read_markdown(path):
        with open(path, "r",encoding = 'utf-8') as f:  # 打开文件
            data = f.read()  # 读取文件
        return data

    readme_text = st.markdown(read_markdown("数据离散化.md"))

    #加载文件与图片

    # 单文件载入
    uploaded_file = st.file_uploader("Choose a file... csv")
    if uploaded_file is not None:
         # To read file as bytes:
         bytes_data = uploaded_file.read()
         st.write(bytes_data)

         # To convert to a string based IO:
         stringio = StringIO(uploaded_file.decode("utf-8"))
         st.write(stringio)

         # To read file as string:
         string_data = stringio.read()
         st.write(string_data)

         # Can be used wherever a "file-like" object is accepted:
         st.write(uploaded_file)
         dataframe = pd.read_csv(uploaded_file)
         st.write(dataframe)

    @st.cache(show_spinner=False)
    def load_local_image(uploaded_file):
        bytes_data = uploaded_file.getvalue()
        image = np.array(Image.open(BytesIO(bytes_data)))
        return image

    uploaded_file = st.sidebar.file_uploader(" ")
    image = load_local_image(uploaded_file)

    #4.3 opencv + yolov3 检测函数
    #
    #
    yolo_boxes = yolo_v3(image, confidence_threshold, overlap_threshold)
    boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels})
    return boxes[["xmin", "ymin", "xmax", "ymax", "labels"]]

def test_one_page(): #单独页面展示

    def sidebar():
        df_parameters = pd.DataFrame({'industry': ['-','金融','医疗'],'name': ['-','A','B'],'time': ['-','5天','7天']})
        select_parameters=[]
        option_industry = '-'
        option_name = '-'
        option_time = '-'

        option_industry = st.sidebar.selectbox('请选择股票行业',df_parameters['industry'])
        if option_industry != '-':
            select_parameters.append(option_industry)
            option_name = st.sidebar.selectbox('请选择股票名称',df_parameters['name'])
        else:
            pass

        if option_name!= '-':
            select_parameters.append(option_name)
            option_time = st.sidebar.selectbox('请选择时间窗口',df_parameters['time'])
        else:
            pass

        if option_time!= '-':
            select_parameters.append(option_time)
            version_option=st.sidebar.selectbox('选择历史版本：',('-','版本1', '版本2', '版本3'))
            st.sidebar.button('切换页面')
            st.sidebar.button('更新参数')

        return select_parameters

    def display():
        st.set_option('deprecation.showPyplotGlobalUse', False)
        #st.title(time.strftime("当前日期：%Y年%m月%d日", time.localtime()))
        start_date = "20200101"
        gap = 365
        date_list = pd.to_datetime(pd.date_range(start=start_date, periods=gap).strftime("%Y%m%d").tolist())
        chart_data = pd.DataFrame(np.random.randn(365, 2),columns=['真实值','预测值'],index=date_list)#传入DataFrame
        plt.plot(date_list,chart_data['真实值'],label='真实值')
        plt.plot(date_list,chart_data['预测值'],label='预测值')
        plt.ylim(bottom=0)
        plt.xticks(rotation=45)
        plt.title(sidebar[0]+'行业'+sidebar[1]+'股票在'+sidebar[2]+'时间窗口下的预测结果',loc='center',fontproperties="SimHei",fontsize=20)
        plt.xlabel('日期',loc='right')
        plt.ylabel('价格',loc='top')
        plt.legend(loc='upper right')
        st.pyplot()

    def button():
        ini_date=datetime.date.today()
        end_date=datetime.date.today()
        st.write('今天的日期：',ini_date)
        st.write('训练数据：')
        ini_date=st.date_input('起始日期：',value=datetime.date(2001,1,1))
        end_date=st.date_input('截止日期：')
        st.button('确定')
        return ini_date,end_date

    sidebar=sidebar()
    #st.write(run)
    if len(sidebar) != 3:
        st.header('请选择参数！')
    else:
        display()
        button()

def test_data_trans(): #做数据转换
    st.write('上传csv文件，进行数据转换 :wave:')
    #file = st.file_uploader('上传文件', type=['csv'], encoding='auto', key=None)
    file = st.file_uploader('上传文件', type=['csv'],key=None)
    @st.cache
    def get_data(file):
        df = pd.DataFrame()
        if file is not None:
            data = []
            for i, line in enumerate(file.getvalue().split('\n')):
                 if i == 0:
                     header = line.split(',')
                 else:
                     data.append(line.split(','))
            df=pd.DataFrame(data,columns=header)
        return df

    df = get_data(file)
    st.write(df)

    ##########y值的行转列
    @st.cache
    def transform_y(df):
        for param_name in df[["PARAM_NAME"]].drop_duplicates().values:
            data_para = df[df["PARAM_NAME"] == param_name[0]]
            final_data = data_para.pivot_table(index=["GLASS_ID", "GLASS_START_TIME", "EQUIP_ID"], columns=["SITE_NAME"],
                                               values=["PARAM_VALUE"], aggfunc=np.sum)
            ncl = [param_name[0] + '_' + str(x + 1) for x in range(final_data.shape[1])]
            final_data = pd.DataFrame(final_data.values, columns=ncl, index=final_data.index)
            final_data = final_data.reset_index()
            final_data["GLASS_ID"] = final_data["GLASS_ID"].fillna(method='ffill')
            final_data.drop_duplicates(subset=["GLASS_ID"], keep='last', inplace=True)
            return final_data

    #####x值的行转列
    @st.cache
    def transform_x(df_x):
        data_x = df_x.pivot_table(index="GLASS_ID", columns="PARAM_NAME", values="PARAM_STR_VALUE",
                                      aggfunc='last')
        return data_x

    #设置button进行调用转换函数
    if st.button("X数据转换"):
        data_x = transform_x(df)
        st.write(data_x)
        data_x.to_csv("data/data_x.csv")
    #if len(df)!=0:
        #data_x=transform_x(df)
    if st.button("X点击下载"):
        data_x = transform_x(df)
        st.write(data_x)
        st.write('./data_x.csv')
    if st.button("Y数据转换"):
        data_y = transform_y(df)
        st.write(data_y)
        data_y.to_csv("data/data_y.csv")
    # if len(df)!=0:
    #     data_y=transform_y(df)
    if st.button("Y点击下载"):
        data_y = transform_y(df)
        st.write(data_y)
        st.write('./data_y.csv')

def test_danamic(): #动态图形展示
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    chart = st.line_chart(last_rows)

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text("%i%% Complete" % i)
        chart.add_rows(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)

    progress_bar.empty()
    st.button("再来一次")

def test_iframe():  #测试跳转新页面

    # 用html方法，将本地的相应的html文件读取出来，赋值给html，然后就可以显示了。
    components.html(
        """
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
        <div id="accordion">
          <div class="card">
            <div class="card-header" id="headingOne">
              <h5 class="mb-0">
                <button class="btn btn-link" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
                手机银行月活客户 #1
                </button>
              </h5>
            </div>
            <div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
              <div class="card-body">
                手机银行客户活跃率 #1
              </div>
            </div>
          </div>
          <div class="card">
            <div class="card-header" id="headingTwo">
              <h5 class="mb-0">
                <button class="btn btn-link collapsed" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
                手机银行app点击率 #2
                </button>
              </h5>
            </div>
            <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
              <div class="card-body">
                手机银行报表制作 #2
              </div>
            </div>
          </div>
        </div>
        """,
        height=600,
    )

    county_list = ['手机银行报表制作','手机银行客户活跃率','手机银行月活客户','手机银行app点击率']
    county_name = st.sidebar.selectbox(
    "请选择：",
    county_list
    )

    #用iframe,可以直接跳转到远程页面。根据选择，展示相应的子页面。
    components.iframe("http://nx.eip.ccb.com/nx/index/index.shtml",height=800,width=800,)

def test_show_html(): #测试根据选择，显示不同的页面
    chart_list = ['pyecharts页面','plotly绘图','pyg2plot绘图','普通页面']
    sidebar = st.sidebar.selectbox(
    "请选择：",
    chart_list
    )

    if sidebar == "普通页面":
        st.title("普通页面")
        url = 'file:///D:/Python/python_src/web/html/temp-plot.html'
        html = urllib.request.urlopen(url).read()
        components.html(html,height=800,width=800,)
    elif sidebar == "pyecharts页面":
        st.title("pyecharts页面")
        url = 'file:///D:/Python/python_src/web/html/pyecharts/tab_base.html'
        html = urllib.request.urlopen(url).read()
        components.html(html,height=800,width=800,)

        url = 'file:///D:/Python/python_src/web/html/pyecharts/overlap_bar_line.html'
        html = urllib.request.urlopen(url).read()
        components.html(html,height=1080,width=1180,)

    elif sidebar == "plotly绘图":
        st.title("plotly绘图")
        url = 'file:///D:/Python/python_src/web/html/plotly/export-image.html'
        html = urllib.request.urlopen(url).read()
        components.html(html,height=800,width=800,)
    elif sidebar == "pyg2plot绘图":
        st.title("pyg2plot绘图")
        url = 'file:///D:/Python/python_src/web/html/pyg2plot/scatter.html'
        html = urllib.request.urlopen(url).read()
        components.html(html,height=800,width=800,)

    else:
        st.title("手机银行报表制作")
        st.write("欢迎使用手机银行报表制作平台")

    #根据选择，展示相应的子页面。
    #components.iframe("http://localhost:8501/html/pyecharts/page_default_layout.html",height=800,width=800,)
    #components.iframe("http://localhost:8501/html/plotly/export-image.html",height=800,width=800,)
    #components.iframe("http://localhost:8501/html/pyg2plot/scatter.html",height=800,width=800,)
    #components.iframe("http://localhost:8501/html/temp-plot.html",height=800,width=800,)

def test_explor_data(): #单页显示数据
    #读取数据库，根据用户的选择而展示不同的结果
    #先读取数据
    @st.cache
    def load_data():
        df = pd.read_excel("./data/客户活跃情况分析明细数据20211019.xls")
        df = df
        df.columns = ['电子渠道', '机构', '频道', '子频道', '访客数', '粘性访客数', '粘性客户数']
        return df

    #给用户选项，首先是选择观察哪一种事件类型。
    df=load_data()
    df

    event_list = df["频道"].unique()
    event_type = st.sidebar.selectbox(    #两个参数：第一个是显示给用户的提示语句，第二个，是选择列表内容。
    "你想看哪一个频道的数据?",
    event_list
    )

    county_list = df["子频道"].unique()
    county_name = st.sidebar.selectbox(
    "哪一个子频道?",
    county_list
    )

    #根据用户选项，提取数据，展示结果。
    #part_df = df[(df["频道"]==event_type) & (df['子频道']==county_name)]
    part_df = df[(df["频道"]==event_type)]
    st.write(f"根据你的筛选，数据包含{len(part_df)}行")

    #展示线性图
    part_df1 = part_df[['访客数','粘性访客数', '粘性客户数']]
    st.line_chart(part_df1)

    #展示柱状图
    st.bar_chart(part_df1)

def test_project(): #一个完整的项目管理案例
    # import streamlit as st
    # import time
    # # 设置网页标题，以及使用宽屏模式
    # st.set_page_config(
    #     page_title="运维管理后台",
    #     layout="wide"

    # )
    # 隐藏右边的菜单以及页脚
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    # 左边导航栏
    sidebar = st.sidebar.radio(
        "导航栏",
        ("首页", "项目管理", "用户管理", "权限管理")
    )
    if sidebar == "项目管理":
        st.title("项目管理")
        # 项目选择框
        project_name = st.selectbox(
            "请选择项目",
            ["项目A", "项目B"]
        )
        if project_name:
            # 表单
            with st.form(project_name):
                project_info_1 = st.text_input("项目信息1", project_name)
                project_info_2 = st.text_input("项目信息2", project_name)
                project_info_3 = st.text_input("项目信息3", project_name)
                submitted = st.form_submit_button("提交")
                if submitted:
                    # 在这里添加真实的业务逻辑
                    # 这是一个进度条
                    bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        bar.progress(i)
                    st.write("项目信息1:%s, 项目信息2:%s, 项目信息3:%s" % (project_info_1, project_info_2, project_info_3))
                    st.success("提交成功")


    elif sidebar == "用户管理":
        st.title("用户管理")
        # 将页面分为左半边和右半边
        left, right = st.beta_columns(2)
        # 左半边页面展示部分
        with left:
            st.header("查看、更新用户信息")
            user_name = st.selectbox(
                "请选择用户",
                ["郑立赛", "乔布斯", "王大拿"]
            )
            if user_name:
                with st.form(user_name):
                    phone_num = st.text_input("手机号", user_name)
                    role = st.multiselect(
                        "用户角色",
                        ["大神", "大拿"],
                        ["大神"]
                    )
                    user_group = st.multiselect(
                        "请选择用户组",
                        ["大神组", "大拿组"],
                        ["大神组"]
                    )
                    submitted = st.form_submit_button("提交")
                    if submitted:
                        # 这里添加真实的业务逻辑
                        st.write("用户名:%s, 手机号:%s, 用户角色:%s, 用户组:%s" % (user_name, phone_num, role, user_group))
                        st.success("提交成功")
        # 右半边页面展示部分
        with right:
            st.header("添加、删除用户")
            user_action = st.selectbox(
                "请选择操作",
                ["添加用户", "删除用户"]
            )
            if user_action:
                with st.form(user_action):
                    if user_action == "添加用户":
                        phone_num = st.text_input("手机号", user_name)
                        role = st.multiselect(
                            "用户角色",
                            ["大神", "大拿"]
                        )
                        user_group = st.multiselect(
                            "请选择用户组",
                            ["大神组", "大拿组"]
                        )
                        submitted = st.form_submit_button("提交")
                        if submitted:
                            # 请在这里添加真实业务逻辑，或者单独写一个业务逻辑函数
                            st.write("user_name:%s, phone_num:%s, role:%s, user_group:%s" % (user_name, phone_num, role, user_group))
                            st.success("提交成功")
                    else:
                        user_group = st.multiselect(
                            "请选择要删除的用户",
                            ["郑立赛", "乔布斯", "王大拿"]
                        )
                        submitted = st.form_submit_button("提交")
                        if submitted:
                            # 请在这里添加真实业务逻辑，或者单独写一个业务逻辑函数
                            st.write("user_name:%s, phone_num:%s, role:%s, user_group:%s" % (user_name, phone_num, role, user_group))
                            st.success("提交成功")
    elif sidebar == "权限管理":
        st.title("权限管理")
        with st.form("auth"):
            user = st.multiselect(
                "选择用户",
                ["郑立赛", "乔布斯", "王大拿"]
            )
            role = st.multiselect(
                "选择用户角色",
                ["大神", "大拿"]
            )
            user_group = st.multiselect(
                "请选择用户组",
                ["大神组", "大拿组"]
            )
            submitted = st.form_submit_button("提交")
            if submitted:
                # 请在这里添加真实业务逻辑，或者单独写一个业务逻辑函数
                st.write(
                    "用户:%s, 角色:%s, 用户组:%s" % (user, role, user_group))
                st.success("提交成功")
    else:
        st.title("运维管理后台")
        st.write("欢迎使用运维管理后台")

def test_fileop(): #测试文件操作
    chart_list = ['查找html文件','查找xls、xlsx文件','查找jpg文件','查找docx文件']
    sidebar = st.sidebar.selectbox(
    "请选择：",
    chart_list
    )

    filename = 'D:/Python/python_src/web/html/temp-plot.html'
    dir = 'D:/Python/python_src/web/'

    def walk_dir(pathName,ext=None): #遍历指定文件夹,返回查找的后缀名文件列表，并打印出目录名、文件名。
        filelist = []
        extlen = len(ext)*(-1)
        for folderName, subfolders, filenames in os.walk(pathName):
            print('\n-----------------------')
            print('当前目录： ' + folderName)
            for subfolder in subfolders:
                print('子目录： ' + folderName + ': ' + subfolder)
            for filename in filenames:
                if ext in filename[extlen:]:
                    filename1 =folderName + '/' + filename
                    filelist.append(filename1)
                print("文件： " + folderName + ': ' + filename)
        return filelist

    def mkdir(path): #判断，生成文件夹
        folder = os.path.exists(path)
        if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
            print("文件夹已创建成功")
        else:
            print("文件夹已存在")

    def makefile(filename): #判断文件是否存在，不存在，创建文件
        if not os.path.exists(filename):
            os.system(r"touch {}".format(path))#调用系统命令行来创建文件

    if sidebar == "查找html文件":
        st.title("查找html文件")
        filelist = walk_dir(dir,ext='html')
        st.markdown("## html文件清单")
        st.write(filelist)

    elif sidebar == "查找xls、xlsx文件":
        st.title("查找xls、xlsx文件")
        filelist = walk_dir(dir,ext='xls')
        st.markdown("## xls文件清单")
        st.write(filelist)
        filelist = walk_dir(dir,ext='xlsx')
        st.markdown("## xlsx文件清单")
        st.write(filelist)
    elif sidebar == "查找jpg文件":
        st.title("查找jpg文件")
        filelist = walk_dir(dir,ext='jpg')
        st.markdown("## jpg文件清单")
        st.write(filelist)
    elif sidebar == "查找docx文件":
        st.title("查找doc、docx文件")
        filelist = walk_dir(dir,ext='doc')
        st.markdown("## doc文件清单")
        st.write(filelist)

        filelist = walk_dir(dir,ext='docx')
        st.markdown("## docx文件清单")
        st.write(filelist)

    else:
        st.title("手机银行报表制作")
        st.write("欢迎使用手机银行报表制作平台")

def test_modi_cfg(): #测试修改配置信息
    #st.write("* 报表配置信息：")
    #侧边栏控件
    #以上控件大部分都有对应的侧边栏形式，如上述st.selectbox若想放置在侧边栏，可使用st.sidebar.selectbox
    # 单选框
    selector1=st.sidebar.selectbox("请选择你喜欢的选项：", [1, 2, 3], key="1")

    # 单选按钮
    selector2=st.sidebar.radio("请选择你喜欢的选项：", [1, 2, 3], key="1")

    # 多选框
    selector = st.sidebar.multiselect("请选择你喜欢的选项：", [1, 2, 3], key="3")
    st.sidebar.write("你的选项分别是：\n",selector1,selector2,selector)

    st.sidebar.subheader('2.选择时间序列')
    #options = np.array(df['日期']).tolist()
    options = ['20160131','20160229','20160331','20160430','20160531','20160630','20160731','20160831','20160930','20161031','20161130','20161231',
                '20170131','20170228','20170331','20170430','20170531','20170630','20170731','20170831','20170930','20171031','20171130','20171231',
                '20180131','20180228','20180331','20180430','20180531','20180630','20180731','20180831','20180930','20181031','20181130','20181231',
                '20190131','20190228','20190331','20190430','20190531','20190630','20190731','20190831','20190930','20191031','20191130','20191231',
                '20200131','20200229','20200331','20200430','20200531','20200630','20200731','20200831','20200930','20201031','20201130','20201231',
                '20210131','20210228','20210331','20210430','20210531','20210630','20210731','20210831','20210930','20211031'
                ]
    start_opt = round(len(options) / 2) - round(len(options) / 4)
    end_opt = round(len(options) / 2) + round(len(options) / 4)

    (start_time, end_time) = st.sidebar.select_slider("请选择时间序列长度：",
         options = options,
         value= (options[start_opt],options[end_opt],),
     )

    st.sidebar.write("时间序列开始时间:",start_time)
    st.sidebar.write("时间序列结束时间:",end_time)


    left, right = st.columns(2)
    # 左半边页面展示部分
    with left:
        config = cfg.get_rpt_conf()
        title = "配置信息："
        msg = " " * 10
        st.title(title)
        st.write(msg)
        st.write(f"当前接收报表的邮箱清单：{config['recivers']}")
        project_name = '修改接收报表的邮箱清单:'

        with st.form(project_name):
            recivers = st.text_area("请输入接收报表的邮箱清单,邮箱之间用分号分隔......", config['recivers'])
            submitted = st.form_submit_button("确认")
            if submitted:
                import time
                config['recivers']= recivers
                cfg.write_conf(config)
                # 这是一个进度条
                bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    bar.progress(i)
                st.success("提交成功")
                config = cfg.get_rpt_conf()
                st.write(f"当前接收报表的邮箱清单：{config['recivers']}")
    # 右半边页面展示部分
    with right:
        # 表单
        # 获取T-2的日期
        time1 = (datetime.date.today() + datetime.timedelta(days=-2)).strftime("%Y%m%d")
        st.write(f'获取T-2的日期:{time1}')
        # 获取T-12的日期
        time2 = (datetime.date.today() + datetime.timedelta(days=-12)).strftime("%Y%m%d")
        poem = '我的孩子，如果可以，我想告诉你世间的一切奥秘，告诉你山川大河，日升日落，光荣和梦想，挫折与悲伤。\
            告诉你，燃料是，点燃自己，照亮别人的东西；火箭是，为了梦想，抛弃自己的东西；生命是，用来燃烧的东西；\
            死亡是，验证生命的东西；宇宙是，让死亡渺小的东西；渺小的尘埃，是宇宙的开始；\
            平凡的渺小，是伟大的开始；而你，我的孩子，是让平凡的我们，想创造新世界的开始。'
        project_name = '手机银行日报表'

        with st.form(project_name):
            project_info_1 = st.text_input("报表名称", project_name)
            number = st.number_input("请输入数字：", 123)
            poem1 = st.text_area("请输入要提交的文本：", poem)
            date1 = st.date_input("请输入日期：")
            x = st.slider("请选择数值范围：", min_value=0, max_value=80)
            start_time = st.slider(
                 "请选择开始日期：",
                 value=datetime.datetime(2021, 1, 1, 9, 30),
                 format="MM/DD/YY - hh:mm")
            # options =['2021-05-31','2016-06-30','2021-01-31','2016-08-30']
            # # options1 = pd.date_range('2021-01-01',periods=365,freq='d')  #生成一个以日为单位的365天的时间列表
            # # options1 = options1.apply(lambda x: str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:])
            # # options = options1.values
            # # print(options)
            # print(time1,time2)
            # (start_time, end_time) = st.select_slider("请选择时间序列长度：",

            #      options = options,
            #      value= (time1,time2,),
            #  )
            # st.write("时间序列开始时间:",end_time)
            # st.write("时间序列结束时间:",start_time)

            submitted = st.form_submit_button("提交")
            if submitted:
                import time
                # 在这里添加真实的业务逻辑
                # 这是一个进度条
                bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    bar.progress(i)
                st.write("报表名称：:%s, 数值:%s, 日期:%s" % (project_info_1, number, date1))
                st.success("提交成功")

def test_download_file(): #测试文件下载功能

    st.markdown("### 点击下载相应文件......")

    with open("E:\Python\python_src\web\data\观云台\9月客户拓展（全国前10）.jpg", "rb") as file:
         btn = st.download_button(
                label="下载图片文件",
                data=file,
                file_name="flower.png",
                mime="image/png"
              )
    with open("E:\Python\python_src\web\data\观云台\访问时段分析明细数据202109.xls", "rb") as file:
         btn = st.download_button(
                label="下载excel文件",
                data=file,
                file_name="访问时段分析明细数据202109.xls",
                mime="application/octet-stream"
              )
    with open("E:\Python\python_src\web\streamli操作手册.docx", "rb") as file:
         btn = st.download_button(
                label="下载docx文件",
                data=file,
                file_name="streamli操作手册.docx",
                help = "streamli操作手册.docx",
                mime="application/octet-stream"
              )



    '''
            st.download_button(label: str, data: Union[str, bytes, TextIO, BinaryIO], file_name: Optional[str] = None, \
            mime: Optional[str] = None, key: Union[str, int, NoneType] = None, help: Optional[str] = None, \
            on_click: Optional[Callable[..., NoneType]] = None, args: Optional[Tuple[Any, ...]] = None, \
            kwargs: Optional[Dict[str, Any]] = None) -> bool

            @st.cache
        ... def convert_df(df):
        ...     return df.to_csv().encode('utf-8')
        >>>
        >>> csv = convert_df(my_large_df)
        >>>
        >>> st.download_button(
        ...     label="下载 CSV",
        ...     data=csv,
        ...     file_name='large_df.csv',
        ...     mime='text/csv',
        ... )

        Download a string as a file:

        >>> text_contents = 'This is some text'
        >>> st.download_button('下载文本 text', text_contents)

        下载二进制文件:

        >>> binary_contents = b'example content'
        >>> # Defaults to 'application/octet-stream'
        >>> st.download_button('下载二进制文件', binary_contents)


        >>> with open("flower.png", "rb") as file:
        ...     btn = st.download_button(
        ...             label="下载图像文件",
        ...             data=file,
        ...             file_name="flower.png",
        ...             mime="image/png"
        ...           )
    '''

def test_pandas_get(): #测试pandas爬虫

    import csv
    chart_list = ['抓取世界大学综合排名','抓取新浪财经基金重仓股数据','抓取证监会披露的IPO数据']
    sidebar = st.sidebar.selectbox(
    "请选择：",
    chart_list
    )

    if sidebar == "抓取世界大学综合排名":
        #抓取世界大学排名（1页数据）
        url1 = 'http://www.compassedu.hk/qs'
        df1 = pd.read_html(url1)[0]  #0表示网页中的第一个Table
        df1.to_csv('世界大学综合排名.csv',index=0)
        st.markdown("## 抓取世界大学综合排名")
        st.dataframe(df1)
    elif sidebar == "抓取新浪财经基金重仓股数据":
        #抓取新浪财经基金重仓股数据（6页数据）
        df2 = pd.DataFrame()
        for i in range(6):
            url2 = 'http://vip.stock.finance.sina.com.cn/q/go.php/vComStockHold/kind/jjzc/index.phtml?p={page}'.format(page=i+1)
            df2 = pd.concat([df2,pd.read_html(url2)[0]])
            print('第{page}页抓取完成'.format(page = i + 1))
        df2.to_csv('./新浪财经数据.csv',encoding='utf-8',index=0)
        st.markdown("## 抓取新浪财经基金重仓股数据")
        st.dataframe(df2)
    elif sidebar == "抓取证监会披露的IPO数据":
        #抓取证监会披露的IPO数据（217页数据）
        from pandas import DataFrame
        import time
        st.markdown("## 抓取证监会披露的IPO数据")
        start = time.time() #计时
        df3 = DataFrame(data=None,columns=['公司名称','披露日期','上市地和板块','披露类型','查看PDF资料']) #添加列名
        for i in range(1,218):
            url3 ='http://eid.csrc.gov.cn/ipo/infoDisplay.action?pageNo=%s&temp=&temp1=&blockType=byTime'%str(i)
            df3_1 = pd.read_html(url3,encoding='utf-8')[2]  #必须加utf-8，否则乱码
            df3_2 = df3_1.iloc[1:len(df3_1)-1,0:-1]  #过滤掉最后一行和最后一列（NaN列）
            df3_2.columns=['公司名称','披露日期','上市地和板块','披露类型','查看PDF资料'] #新的df添加列名
            df3 = pd.concat([df3,df3_2])  #数据合并
            st.write('第{page}页抓取完成'.format(page=i))
        df3.to_csv('./上市公司IPO信息.csv', encoding='utf-8',index=0) #保存数据到csv文件
        end = time.time()
        st.write('共抓取',len(df3),'家公司,' + '用时',round((end-start)/60,2),'分钟')
        st.dataframe(df3)
    else:
        st.title("Pandas抓取数据")
        st.write("欢迎使用功能强大的Pandas")

def test_datetime_filter(): #对时间序列数据集进行可视化过滤

    def df_filter(message,df):

        slider_1, slider_2 = st.slider('* %s' % (message),0,len(df)-1,[0,len(df)-1],1)  #滑块将返回两个值，即开始日期时间和结束日期时间值

        while len(str(df.iloc[slider_1][1]).replace('.0','')) < 4:
            #从我们的开始/结束时间列中删除任何后面的小数点位，并在时间少于一个小时的情况下添加前面的零，即12:00AM引用为0
            df.iloc[slider_1,1] = '0' + str(df.iloc[slider_1][1]).replace('.0','')

        while len(str(df.iloc[slider_2][1]).replace('.0','')) < 4:
            df.iloc[slider_2,1] = '0' + str(df.iloc[slider_1][1]).replace('.0','')

        #将日期添加到时间中，并以使用datetime可以理解的格式解析我们的datetime
        start_date = datetime.datetime.strptime(str(df.iloc[slider_1][0]).replace('.0','') + str(df.iloc[slider_1][1]).replace('.0',''),'%Y/%m/%d%H%M%S')
        #使用strftime函数来重新格式化开始/结束
        start_date = start_date.strftime('%Y年%m月%d日, %I:%M%p')

        end_date = datetime.datetime.strptime(str(df.iloc[slider_2][0]).replace('.0','') + str(df.iloc[slider_2][1]).replace('.0',''),'%Y/%m/%d%H%M%S')
        end_date = end_date.strftime('%Y年%m月%d日, %I:%M%p')

        st.info('开始: **%s** 结束: **%s**' % (start_date,end_date))

        #将过滤后的索引应用到我们的数据集
        filtered_df = df.iloc[slider_1:slider_2+1][:].reset_index(drop=True)

        return filtered_df

    def download_csv(name,df):
        csv = df.to_csv(index=False)
        base = base64.b64encode(csv.encode()).decode()
        file = (f'<a href="data:file/csv;base64,{base}" download="%s.csv">下载文件</a>' % (name))
        return file

    df = pd.read_csv('./data/vega_datasets/file_path.csv')
    st.markdown('* 原始数据')
    st.write(df)

    st.title('日期时间筛选器')
    filtered_df = df_filter('移动滑块，筛选数据',df)

    column_1, column_2 = st.columns(2)

    with column_1:
        st.title('数据')
        st.write(filtered_df)

    with column_2:
        st.title('图表')
        st.line_chart(filtered_df['value'])

    st.markdown(download_csv('筛选后的数据集',filtered_df),unsafe_allow_html=True)

def main():  #主程序模块，调用其他程序模块
    st.sidebar.markdown("# 主程序导航图")

    #
    options = ['基本功能展示','write功能展示','动态表格展示','控件功能展示','绘图、图片功能展示',\
                '音频、视频功能展示','网页布局','图像目标检测demo网页','单独页面展示','数据转换',\
                '动态图形展示','跳转新页面','展示HTML文件内容','单页显示数据','项目管理','文件查找',\
                '修改配置信息','文件下载','图像识别','altair可视化数据分析','对时间序列数据集进行可视化过滤']
    object_type = st.sidebar.selectbox("请选择程序模块", options, 1)
    # min_elts, max_elts = st.sidebar.slider("多少 %s     (选择一个范围)?" % object_type, 0, 25, [10, 20])
    # selected_frame_index = st.sidebar.slider("选择一帧 (帧的索引)", 0, len(object_type) - 1, 0)
    st.sidebar.markdown("------")
    st.sidebar.markdown("## 子模块  ")
    st.sidebar.markdown("---")


    if object_type == '动态表格展示':
        test_table()
    elif object_type == '基本功能展示':
        test_base()
    elif object_type == 'write功能展示':
        test_write()
    elif object_type == '控件功能展示':
        test_control()
    elif object_type == '绘图、图片功能展示':
        test_plot()
    elif object_type == '音频、视频功能展示':
        test_audio_video()
    elif object_type == '网页布局':
        test_explor()
    elif object_type == '图像目标检测demo网页':
        test_pic()
    elif object_type == '单独页面展示':
        test_one_page()
    elif object_type == '数据转换':
        test_data_trans()
    elif object_type == '动态图形展示':
        test_danamic()
    elif object_type == '跳转新页面':
        test_iframe()
    elif object_type == '展示HTML文件内容':
        test_show_html()
    elif object_type == '单页显示数据':
        test_explor_data()
    elif object_type == '项目管理':
        test_project()
    elif object_type == '文件查找':
        test_fileop()
    elif object_type == '修改配置信息':
        test_modi_cfg()
    elif object_type == '文件下载':
        test_download_file()
    elif object_type == '图像识别':
        tpc.main()
    elif object_type == 'altair可视化数据分析':
        alt_app.main()
    elif object_type == 'pandas爬虫':
        test_pandas_get() #测试pandas爬虫
    elif object_type == '对时间序列数据集进行可视化过滤':
        test_datetime_filter() #对时间序列数据集进行可视化过滤

if __name__ == '__main__':
    main()
