###
#用Altair进行数据可视化分析：
#altair 可以绘制常见的基础图，如“bar”, “circle”, “square”, “tick”,“line”, * “area”, “point”, “rule”, “geoshape”, and “text”等
#本程序测试altair绘制图形的各种方法。
#Altair 只需要三个主要的参数：
'''
Mark. 数据在图形中的表达形式。点、线、柱状还是圆圈？
Channels. 决定什么数据应该作为x轴，什么作为y轴；图形中数据标记的大小和颜色。
Encoding. 指定数据变量类型。日期变量(T)、量化变量(Q)还是类别变量(N)？

基于以上三个参数，Altair 将会选择合理的默认值来显示我们的数据。
Altair 最让人着迷的地方是，它能够合理的选择颜色。如果我们在 Encoding 中指定变量类型为量化变量，
那么 Altair 将会使用连续的色标来着色（默认为 浅蓝色-蓝色-深蓝色）。如果变量类型指定为类别变量，那么 Altair 会为每个类别赋予不同的颜色。
（例如 红色，黄色，蓝色）

'''

import streamlit as st
import pandas as pd
import altair as alt
from altair import datum
#import keras
from PIL import Image
import numpy as np
#from keras.utils.vis_utils import plot_model
#import seaborn as sns

#from vega_datasets import data

def alt_sigle(): #简单绘图

    data = pd.DataFrame({'country_id': [1, 2, 3, 4, 5, 6],
                         'population': [1, 100, 200, 300, 400, 500],
                         'income':     [1000, 50, 200, 300, 200, 150]})

    #首先我们绘制每个国家的人口数据：
    categorical_chart = alt.Chart(data).mark_circle(size=200).encode(
                        x='population:Q',
                        color='country_id:Q')
    st.altair_chart(categorical_chart)

    #从上图可以看出，Altair 选择了连续色标，在本例中这是没有意义的。
    #问题的根源在于，我们将 country_id 定义为量化变量，而实际上，它应该是一个类别变量，修改代码如下：
    # We changed color='country_id:Q' to color='country_id:N' to indicate it is a nominal variable
    categorical_chart = alt.Chart(data).mark_circle(size=200).encode(
                            x='population:Q',
                            color='country_id:N')
    st.altair_chart(categorical_chart)
    #从图中可以看到，每个国家都用了不同的颜色表示。我们仅仅改变了变量 country_id 的编码，
    #即用 N (Nominal 名义变量)替换了 Q (Quantitative 量化变量)。
    #这点小小的改变就足以使得 Altair 明白，它不该使用连续色标，而是使用独立色标。


    #Altair 的另一个美妙之处就是，我们可以从现有的图表中创建新的图表。
    #例如，我们现在要加入新的数据 income，我们唯一需要做的就是告诉 Altair：用 income 作为y轴，代码如下所示：
    categorical_chart = alt.Chart(data).mark_circle(size=200).encode(
                        x='population:Q',
                        y='income:Q',
                        color='country_id:N')
    st.altair_chart(categorical_chart)

    #如果想添加数据提示的功能（tooltip，鼠标悬停在数据上时，会显示该数据的详细信息），只需要增加一行代码：
    categorical_chart = alt.Chart(data).mark_circle(size=200).encode(
                      x='population:Q',
                      y='income:Q',
                      color='country_id:N',
                      tooltip=['country_id', 'population', 'income'])
    st.altair_chart(categorical_chart)


    #最简单的画条形图
    data = alt.Data(values=[{'x': 'A', 'y': 5},
                            {'x': 'B', 'y': 3},
                            {'x': 'C', 'y': 6},
                            {'x': 'D', 'y': 7},
                            {'x': 'E', 'y': 2}])
    bar1 = alt.Chart(data).mark_bar().encode(
        x='x:O',  # specify ordinal data
        y='y:Q',  # specify quantitative data
    )
    st.altair_chart(bar1)

    #使用日期型索引作为X轴
    rand = np.random.RandomState(0)

    data = pd.DataFrame({'value': rand.randn(100).cumsum()},
                        index=pd.date_range('2018', freq='D', periods=100))

    #if you would like the index to be available to the chart, you can explicitly turn it into a column 
    #using the reset_index() method of Pandas dataframes:

    line1 = alt.Chart(data.reset_index()).mark_line().encode(
        x='index:T',
        y='value:Q'
    )
    st.altair_chart(line1)

    base = alt.Chart(data.reset_index()).mark_bar().encode(
        alt.Y('mean(value):Q', title='total population')
    ).properties(
        width=200,
        height=200
    )

    chart1 = alt.hconcat(
        base.encode(x='index:Q').properties(title='year=quantitative'),
        base.encode(x='index:O').properties(title='year=ordinal')
    )
    st.altair_chart(chart1)

def alt_iris(): ##快速绘图
    '''
    绘图步骤拆分 :

    由alt.Chart(pd_iris).mark_point().encode(x='sepalLength',y='sepalWidth',color='species')这段代码可知，Altair绘图
    主要用到Chart()方法、mark_*()方法、和encode()方法。

    Chart()方法将数据转化为altair.vegalite.v4.api.Chart对象

    括号内可设置图像的高度、宽度、背景色等等，
    详细见：https://altair-viz.github.io/user_guide/generated/toplevel/altair.Chart.html?highlight=chart

    mark_*()方法指定要展示的图形，例如绘制散点图mark_point()
    mark_*()方法设置图形属性，如颜色color、大小size等
    括号内可设置待展示图形的各种属性，以mark_point()设置点颜色为例如下。
    alt.Chart(pd_iris).mark_point(color='firebrick',size=48)

    encode()方法设置坐标轴的映射

    '''

    filename = './data/vega_datasets/iris.json'
    pd_iris = pd.read_json(filename)
    st.write(pd_iris.head(20))

    mplt = alt.Chart(pd_iris).mark_point(color='firebrick',size=48).encode(x='sepalLength',
                                           y='sepalWidth',
                                           color='species')

    st.altair_chart(mplt)

def alt_cars(): #更复杂的图形
    '''
    更复杂的图形，如个性化分面图标题、图例、会用到configure_*()方法、selection()方法、condition()方法、binding_*()方法、

    configure_*()方法个性化图像属性
    configure_header()方法个性化header

    configure_legend()方法个性化图例

    更多configure类方法介绍见：https://altair-viz.github.io/user_guide/configuration.html

    selection、condition、binding使得altair图形能和鼠标更好交互
    这里主要用到selection()、condition()、binding()方法，简单介绍，详细见：https://altair-viz.github.io/user_guide/interactions.html

    selection()方法
    鼠标可以轻捕捉图形某一部分。
    condition()方法
    让鼠标捕捉的部分高亮，未捕捉的部分暗淡。

    '''

    #source = data.cars.url
    filename = './data/vega_datasets/cars.json'
    source = pd.read_json(filename)
    st.markdown('### 源数据')
    st.write(source.head(20))

    ####### configure_header
    st.markdown('### configure_header的效果')
    chart = alt.Chart(source).mark_point().encode(x='Horsepower:Q',
                                                  y='Miles_per_Gallon:Q',
                                                  color='Origin:N',
                                                  column='Origin:N').properties(
                                                      width=180, height=180)
    chart.configure_header(titleColor='green',
                           titleFontSize=14,
                           labelColor='red',
                           labelFontSize=14)
    st.altair_chart(chart)

    ####### configure_legend
    st.markdown('### configure_legend的效果')
    chart = alt.Chart(source).mark_point().encode(x='Horsepower:Q',
                                              y='Miles_per_Gallon:Q',
                                              color='Origin:N')

    chart.configure_legend(strokeColor='gray',
                           fillColor='#EEEEEE',
                           padding=10,
                           cornerRadius=10,
                           orient='top-right')
    st.altair_chart(chart)

    ####### selection、condition()方法
    st.markdown('### selection、condition()方法的效果')
    brush = alt.selection_interval()
    # chart = alt.Chart(source).mark_point().encode(x='Horsepower:Q',
    #                                           y='Miles_per_Gallon:Q',
    #                                           color='Origin:N').add_selection(brush)
    chart = alt.Chart(source).mark_point().encode(x='Horsepower:Q',
                                              y='Miles_per_Gallon:Q',
                                              color=alt.condition(brush,'Origin:N',alt.value('gray'))).add_selection(brush)
    chart.configure_legend(strokeColor='gray',
                           fillColor='#EEEEEE',
                           padding=10,
                           cornerRadius=10,
                           orient='top-right')
    st.altair_chart(chart)

    ####### selection、condition()、binding_*()方法
    st.markdown('### selection、condition()、binding_*()方法方法的效果')
    input_dropdown = alt.binding_select(options=['Europe','Japan','USA'])
    selection =alt.selection_single(fields=['Origin'],bind=input_dropdown,name='Country of')
    color = alt.condition(selection,alt.Color('Origin:N',legend=None),alt.value('lightgray'))

    chart = alt.Chart(source).mark_point().encode(
                                            x='Horsepower:Q',
                                            y='Miles_per_Gallon:Q',
                                            color=color,
                                            tooltip='Name:N').add_selection(selection)
    st.altair_chart(chart)

    #因为color参数的类型不同，生成不同类型的图表
    st.markdown('### 因为color参数的类型不同，产生不同的效果')
    base = alt.Chart(source).mark_point().encode(
        x='Horsepower:Q',
        y='Miles_per_Gallon:Q',
    ).properties(
        width=150,
        height=150
    )

    chart = alt.vconcat(
       base.encode(color='Cylinders:Q').properties(title='quantitative'),
       base.encode(color='Cylinders:O').properties(title='ordinal'),
       base.encode(color='Cylinders:N').properties(title='nominal'),
    )
    st.altair_chart(chart)


def alt_complex(): #构建复合图形
    '''
    Layer, HConcat, VConcat, Repeat, Facet助力altair轻松构建复合图形
    '''
    #hconcat水平方向拼图
    filename = './data/vega_datasets/iris.json'
    iris = pd.read_json(filename)

    chart1 = alt.Chart(iris).mark_point().encode(x='petalLength:Q',
                                                 y='petalWidth:Q',
                                                 color='species:N').properties(
                                                     height=300, width=300)

    chart2 = alt.Chart(iris).mark_bar().encode(x='count()',
                                               y=alt.Y('petalWidth:Q',
                                                       bin=alt.Bin(maxbins=30)),
                                               color='species:N').properties(
                                                   height=300, width=100)
    #st.altair_chart(chart1 | chart2)
    #或
    cp_chart = alt.hconcat(chart1, chart2)
    st.altair_chart(cp_chart)

    #vconcat垂直方向拼图
    filename = './data/vega_datasets/stocks.csv'
    source = pd.read_csv(filename)
    st.write(source.head(20))

    brush = alt.selection(type='interval',encodings = ['x'])
    base = alt.Chart(source).mark_area().encode(
        x='date:T',
        y='price:Q').properties(width=600,height=200)
    upper = base.encode(alt.X('date:T',scale = alt.Scale(domain=brush)))
    lower = base.properties(height=60).add_selection(brush)

    cp_chart1 = alt.vconcat(upper,lower)
    st.altair_chart(cp_chart1)

    #LayerChart图层叠加
    #之一：
    cp_chart2 = alt.layer(base.mark_line(),base.mark_point(),base.mark_rule()).interactive()
    st.altair_chart(cp_chart2)

    #RepeatChart绘制类似图形
    base = alt.Chart().mark_point().encode(color='species:N').properties(
        width=200, height=200).interactive()

    chart = alt.vconcat(data=iris)
    for y_encoding in ['petalLength:Q', 'petalWidth:Q']:
        row = alt.hconcat()
        for x_encoding in ['sepalLength:Q', 'sepalWidth:Q']:
            row |= base.encode(x=x_encoding, y=y_encoding)
        chart &= row
    #chart
    st.altair_chart(chart)

    #FacetChart图形分面
    from altair.expr import datum
    base = alt.Chart(iris).mark_point().encode(x='petalLength:Q',
                                               y='petalWidth:Q',
                                               color='species:N').properties(
                                                   width=160, height=160)

    chart = alt.hconcat()
    for species in ['setosa', 'versicolor', 'virginica']:
        chart |= base.transform_filter(datum.species == species)
    st.altair_chart(chart)

    #Chart.resolve_scale(), Chart.resolve_axis(), and Chart.resolve_legend()个性化复合图形
    #例如，使用resolve_scale()分别给两个图使用颜色盘。
    filename = './data/vega_datasets/cars.json'
    source = pd.read_json(filename)

    base = alt.Chart(source).mark_point().encode(
        x='Horsepower:Q', y='Miles_per_Gallon:Q').properties(width=200, height=200)

    chart =  alt.concat(base.encode(color='Origin:N'),
               base.encode(color='Cylinders:O')).resolve_scale(color='independent')
    st.altair_chart(chart)

def alt_weather(): #综合案例：分析天气数据

    filename = './data/vega_datasets/seattle-weather.csv'
    df = pd.read_csv(filename)
    st.write(df.head(20))

    scale = alt.Scale(
        domain=['sun', 'fog', 'drizzle', 'rain', 'snow'],
        range=['#e7ba52', '#c7c7c7', '#aec7e8', '#1f77b4', '#9467bd'])
    brush = alt.selection(type='interval')
    points = alt.Chart().mark_point().encode(
        alt.X('temp_max:Q', title='Maximum Daily Temperature (C)'),
        alt.Y('temp_range:Q', title='Daily Temperature Range (C)'),
        color=alt.condition(brush,
                            'weather:N',
                            alt.value('lightgray'),
                            scale=scale),
        size=alt.Size('precipitation:Q',
                      scale=alt.Scale(range=[1, 200]))).transform_calculate(
                          "temp_range",
                          "datum.temp_max - datum.temp_min").properties(
                              width=600, height=400).add_selection(brush)

    bars = alt.Chart().mark_bar().encode(
        x='count()',
        y='weather:N',
        color=alt.Color('weather:N', scale=scale),
    ).transform_calculate(
        "temp_range",
        "datum.temp_max - datum.temp_min").transform_filter(brush).properties(
            width=600)

    mplt = alt.vconcat(points, bars, data=df)
    st.altair_chart(mplt)

def alt_chart1():
    st.markdown('## altair 画热力图')
    x, y = np.meshgrid(range(-5, 5), range(-5, 5))
    z = x ** 2 + y ** 2

    # Convert this grid to columnar data expected by Altair
    source = pd.DataFrame({'x': x.ravel(),
                         'y': y.ravel(),
                         'z': z.ravel()})

    st.write(source)
    chart1 = alt.Chart(source).mark_rect().encode(
        x='x:O',
        y='y:O',
        color='z:Q'
    )
    st.altair_chart(chart1)

    st.markdown('## altair 画简单线图')
    x = np.arange(100)
    source = pd.DataFrame({
      'x': x,
      'f(x)': np.sin(x / 5)
    })

    st.markdown('---')
    st.write(source)
    chart1 =  alt.Chart(source).mark_line().encode(
        x='x',
        y='f(x)'
    )
    st.altair_chart(chart1)

    st.markdown('## altair 画Strip Plo线图')
    filename = './data/vega_datasets/cars.json'
    source = pd.read_json(filename)

    st.markdown('---')
    st.write(source)
    chart1 =  alt.Chart(source).mark_tick().encode(
        x='Horsepower:Q',
        y='Cylinders:O'
    )
    st.altair_chart(chart1)

    st.markdown('## altair 画带阴影的折线图')
    line = alt.Chart(source).mark_line().encode(
        x='Year',
        y='mean(Miles_per_Gallon)'
    )

    band = alt.Chart(source).mark_errorband(extent='ci').encode(
        x='Year',
        y=alt.Y('Miles_per_Gallon', title='Miles/Gallon'),
    )

    chart1 =  band + line
    st.altair_chart(chart1)


    st.markdown('## altair 画柱形图，特殊标准颜色,带标识')
    filename = './data/vega_datasets/wheat.json'
    source = pd.read_json(filename)

    st.markdown('---')
    st.write(source)
    bars =  alt.Chart(source).mark_bar().encode(
        x='year:O',
        y="wheat:Q",
        # The highlight will be set on the result of a conditional statement
        color=alt.condition(
            alt.datum.year == 1745,  # If the year is 1810 this test returns True,
            alt.value('yellow'),     # which sets the bar orange.
            alt.value('orange')   # And if it's not true it sets the bar steelblue.
        )
    ).properties(width=600)

    text = bars.mark_text(
        align='left',
        baseline='middle',
        dx=3  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(
        text='wheat:Q'
    )
    chart1 =  (bars + text).properties(height=900)
    st.altair_chart(chart1)

    st.markdown('## altair 画柱形图，带平均线')
    rule = alt.Chart(source).mark_rule(color='red').encode(
        y='mean(wheat):Q'
    )
    chart1 =  (bars + rule).properties(height=900)
    st.altair_chart(chart1)

    st.markdown('## altair 画柱形图加折线')
    base = alt.Chart(source).encode(x='year:O')
    bar = base.mark_bar().encode(y='wheat:Q')
    line =  base.mark_line(color='red').encode(
        y='wages:Q'
    )

    chart1 =  (bar + line).properties(width=600)
    st.altair_chart(chart1)

    st.markdown('## altair 画柱形图，改变X轴坐标、标识')
    source = pd.DataFrame({'Activity': ['Sleeping', 'Eating', 'TV', 'Work', 'Exercise'],
                           'Time': [8, 2, 4, 8, 2]})

    chart1 =  alt.Chart(source).transform_joinaggregate(
        TotalTime='sum(Time)',
    ).transform_calculate(
        PercentOfTotal="datum.Time / datum.TotalTime"
    ).mark_bar().encode(
        alt.Y('PercentOfTotal:Q', axis=alt.Axis(format='.0%')),
        x='Activity:N'
    )
    st.altair_chart(chart1)

    st.markdown('## altair 画分组柱形图')
    filename = './data/vega_datasets/barley.json'
    source = pd.read_json(filename)
    st.dataframe(source)


    chart1 =  alt.Chart(source).mark_bar().encode(
        x='year:O',
        y='sum(yield):Q',
        color='year:N',
        column='site:N'
    )
    st.altair_chart(chart1)

    st.markdown('## altair 画水平堆叠柱形图')
    chart1 =  alt.Chart(source).mark_bar().encode(
        x='sum(yield)',
        y='variety',
        color='site'
    )
    st.altair_chart(chart1)

    st.markdown('## altair 画水平堆叠柱形图，带数值显示')
    bars = alt.Chart(source).mark_bar().encode(
        x=alt.X('sum(yield):Q', stack='zero'),
        y=alt.Y('variety:N'),
        color=alt.Color('site')
    )

    text = alt.Chart(source).mark_text(dx=-15, dy=3, color='white').encode(
        x=alt.X('sum(yield):Q', stack='zero'),
        y=alt.Y('variety:N'),
        detail='site:N',
        text=alt.Text('sum(yield):Q', format='.1f')
    )

    chart1 =  bars + text
    st.altair_chart(chart1)

    st.markdown('## altair 画多系列折线图')
    filename = './data/vega_datasets/stocks.csv'
    source = pd.read_csv(filename)
    st.dataframe(source)

    chart1 =  alt.Chart(source).mark_line().encode(
        x='date',
        y='price',
        color='symbol',
        strokeDash='symbol',
    )
    st.altair_chart(chart1)

    st.markdown('## altair 画带拟合曲线的散点图')
    #Polynomial Fit Plot with Regression Transform
    # Generate some random data
    rng = np.random.RandomState(1)
    x = rng.rand(40) ** 2
    y = 10 - 1.0 / (x + 0.1) + rng.randn(40)
    source = pd.DataFrame({"第x年": x, "增长率": y})
    st.markdown('---')
    st.write(source)

    # Define the degree of the polynomial fits
    degree_list = [1, 3, 5]

    base = alt.Chart(source).mark_circle(color="black").encode(
            alt.X("第x年"), alt.Y("增长率")
    ).properties(
        width=650
    )

    polynomial_fit = [
        base.transform_regression(
            "第x年", "增长率", method="poly", order=order, as_=["第x年", str(order)]
        )
        .mark_line()
        .transform_fold([str(order)], as_=["degree", "增长率"])
        .encode(alt.Color("degree:N"))
        for order in degree_list
    ]

    chart1 = alt.layer(base, *polynomial_fit)
    st.altair_chart(chart1)

    ##########################
    st.markdown('## altair 画多条折线图（通过transform_fold方法）')
    rand = np.random.RandomState(0)
    data = pd.DataFrame({
        'date': pd.date_range('2019-01-01', freq='D', periods=30),
        'A': rand.randn(30).cumsum(),
        'B': rand.randn(30).cumsum(),
        'C': rand.randn(30).cumsum(),
    })
    st.markdown('---')
    st.write(data)

    chart1 = alt.Chart(data).transform_fold(
        ['A', 'B', 'C'],
    ).mark_line().encode(
        x='date:T',
        y='value:Q',
        color='key:N'
    )
    st.altair_chart(chart1)

    #############################
    st.markdown('## altair 画带线性回归的散点图（通过transform_regression方法）')
    np.random.seed(42)
    x = np.linspace(0, 10)
    y = x - 5 + np.random.randn(len(x))

    df = pd.DataFrame({'x': x, 'y': y})
    st.markdown('---')
    st.write(df)

    chart = alt.Chart(df).mark_point().encode(
        x='x',
        y='y'
    )

    chart1 = chart + chart.transform_regression('x', 'y').mark_line()
    st.altair_chart(chart1)

    ####################
    st.markdown('## altair 通过transform_window方法分组排序画图')
    filename = './data/vega_datasets/movies.csv'
    movies = pd.read_csv(filename)
    st.write(movies)

    chart1 = alt.Chart(movies).transform_window(
        sort=[{'field': 'IMDB_Rating'}],
        frame=[None, 0],
        cumulative_count='count(*)',
    ).mark_area().encode(
        x='IMDB_Rating:Q',
        y='cumulative_count:Q',
    )
    st.altair_chart(chart1)

def alt_chart2():

    st.markdown('## altair 画日期序列类型图')
    filename = './data/vega_datasets/seattle-temps.csv'
    temps = pd.read_csv(filename)
    st.write(temps)

    chart2 = alt.Chart(temps).mark_line().encode(
        x='date:T',
        y='temp:Q'
    )
    st.altair_chart(chart2)

    #################(进行日期转换、数据计算)##
    chart2 = alt.Chart(temps).mark_line().encode(
        x='month(date):T',
        y='mean(temp):Q'
    )
    st.altair_chart(chart2)
    ###############
    chart2 = alt.Chart(temps).mark_bar().encode(
        x='month(date):O',
        y='mean(temp):Q'
    )
    st.altair_chart(chart2)

    ###############
    chart2 = alt.Chart(temps).mark_rect().encode(
        alt.X('date(date):O', title='day'),
        alt.Y('month(date):O', title='month'),
        color='max(temp):Q'
    ).properties(
        title="2010 Daily High Temperatures in Seattle (F)"
    )
    st.altair_chart(chart2)

    #################
    chart2 = alt.Chart(temps).mark_line().encode(
        alt.X('month:T', axis=alt.Axis(format='%b')),
        y='mean(temp):Q'
    ).transform_timeunit(
        month='month(date)'
    )
    st.altair_chart(chart2)

    st.markdown('------')
    st.markdown('## altair 画透视图类型图')
    df = pd.DataFrame.from_records([
        {"country": "Norway", "type": "gold", "count": 14},
        {"country": "Norway", "type": "silver", "count": 14},
        {"country": "Norway", "type": "bronze", "count": 11},
        {"country": "Germany", "type": "gold", "count": 14},
        {"country": "Germany", "type": "silver", "count": 10},
        {"country": "Germany", "type": "bronze", "count": 7},
        {"country": "Canada", "type": "gold", "count": 11},
        {"country": "Canada", "type": "silver", "count": 8},
        {"country": "Canada", "type": "bronze", "count": 10}
    ])

    st.markdown('---')
    st.write(df)

    chart2 = alt.Chart(df).transform_pivot(
        'type',
        groupby=['country'],
        value='count'
    ).mark_bar().encode(
        x='gold:Q',
        y='country:N',
    )
    st.altair_chart(chart2)

    ##############use pivot to create a single tooltip for values on multiple lines:
    filename = './data/vega_datasets/stocks.csv'
    source = pd.read_csv(filename)
    st.markdown('---')
    st.write(source)

    base = alt.Chart(source).encode(x='date:T')
    columns = sorted(source.symbol.unique())
    selection = alt.selection_single(
        fields=['date'], nearest=True, on='mouseover', empty='none', clear='mouseout'
    )

    lines = base.mark_line().encode(y='price:Q', color='symbol:N')
    points = lines.mark_point().transform_filter(selection)

    rule = base.transform_pivot(
        'symbol', value='price', groupby=['date']
    ).mark_rule().encode(
        opacity=alt.condition(selection, alt.value(0.3), alt.value(0)),
        tooltip=[alt.Tooltip(c, type='quantitative') for c in columns]
    ).add_selection(selection)

    chart2 = lines + points + rule
    #chart2.save('./data/stocks.svg', scale_factor=2.0)
    chart2.save('./html/stocks.html', embed_options={'renderer':'svg'})
    st.altair_chart(chart2)


    st.markdown('------')
    st.markdown('## altair 画多种互动类型的图')
    ####################Multiple Interactions
    #Dropdown: Filters the movies by genre
    #Radio Buttons: Highlights certain films by Worldwide Gross
    #Mouse Drag and Scroll: Zooms the x and y scales to allow for panning.

    #st.markdown('* 下载movies数据......')
    # movies = alt.UrlData(
    #     data.movies.url,
    #     format=alt.DataFormat(parse={"Release_Date":"date"})
    # )

    # movies = pd.read_json('https://cdn.jsdelivr.net/npm/vega-datasets@v1.29.0/data/movies.json')
    # #movies.rename(columns={'Release_Date':'date'},inplace=True)
    # movies.to_csv('./data/vega_datasets/movies.csv',index = False)

    filename = './data/vega_datasets/movies.csv'
    movies = pd.read_csv(filename)


    ratings = ['G', 'NC-17', 'PG', 'PG-13', 'R']
    genres = ['Action', 'Adventure', 'Black Comedy', 'Comedy',
           'Concert/Performance', 'Documentary', 'Drama', 'Horror', 'Musical',
           'Romantic Comedy', 'Thriller/Suspense', 'Western']

    base = alt.Chart(movies, width=200, height=200).mark_point(filled=True).transform_calculate(
        Rounded_IMDB_Rating = "floor(datum.IMDB_Rating)",
        Hundred_Million_Production =  "datum.Production_Budget > 100000000.0 ? 100 : 10",
        Release_Year = "year(datum.Release_Date)"
    ).transform_filter(
        alt.datum.IMDB_Rating > 0
    ).transform_filter(
        alt.FieldOneOfPredicate(field='MPAA_Rating', oneOf=ratings)
    ).encode(
        x=alt.X('Worldwide_Gross:Q', scale=alt.Scale(domain=(100000,10**9), clamp=True)),
        y='IMDB_Rating:Q',
        tooltip="Title:N"
    ).properties(
        width=350
    )

    # A slider filter
    year_slider = alt.binding_range(min=1969, max=2018, step=1)
    slider_selection = alt.selection_single(bind=year_slider, fields=['Release_Year'], name="Release Year_")


    filter_year = base.add_selection(
        slider_selection
    ).transform_filter(
        slider_selection
    ).properties(title="Slider Filtering")

    # A dropdown filter
    genre_dropdown = alt.binding_select(options=genres)
    genre_select = alt.selection_single(fields=['Major_Genre'], bind=genre_dropdown, name="Genre")

    filter_genres = base.add_selection(
        genre_select
    ).transform_filter(
        genre_select
    ).properties(title="Dropdown Filtering")

    #color changing marks
    rating_radio = alt.binding_radio(options=ratings)

    rating_select = alt.selection_single(fields=['MPAA_Rating'], bind=rating_radio, name="Rating")
    rating_color_condition = alt.condition(rating_select,
                          alt.Color('MPAA_Rating:N', legend=None),
                          alt.value('lightgray'))

    highlight_ratings = base.add_selection(
        rating_select
    ).encode(
        color=rating_color_condition
    ).properties(title="Radio Button Highlighting")

    # Boolean selection for format changes
    input_checkbox = alt.binding_checkbox()
    checkbox_selection = alt.selection_single(bind=input_checkbox, name="Big Budget Films")

    size_checkbox_condition = alt.condition(checkbox_selection,
                                            alt.SizeValue(25),
                                            alt.Size('Hundred_Million_Production:Q')
                                           )

    budget_sizing = base.add_selection(
        checkbox_selection
    ).encode(
        size=size_checkbox_condition
    ).properties(title="Checkbox Formatting")

    st.write(movies)

    chart2 = ( filter_year | filter_genres) &  (highlight_ratings | budget_sizing  )
    st.altair_chart(chart2)


    ###########################
    st.markdown('------')
    st.markdown('## altair 画人口指标逐年对比图')

    #st.markdown('* 下载population数据......')
    #source = data.population.url

    # source = pd.read_json('https://cdn.jsdelivr.net/npm/vega-datasets@v1.29.0/data/population.json')
    # source.to_csv('./data/vega_datasets/population.csv',index = False)

    filename = './data/vega_datasets/population.json'
    source = pd.read_json(filename)

    #source = './data/vega_datasets/population.json'
    slider = alt.binding_range(min=1850, max=2000, step=10)
    select_year = alt.selection_single(name='year', fields=['year'],
                                       bind=slider, init={'year': 1960})

    base = alt.Chart(source).add_selection(
        select_year
    ).transform_filter(
        select_year
    ).transform_calculate(
        gender=alt.expr.if_(alt.datum.sex == 1, 'Male', 'Female')
    ).properties(
        width=350
    )

    color_scale = alt.Scale(domain=['Male', 'Female'],
                            range=['#1f77b4', '#e377c2'])

    left = base.transform_filter(
        alt.datum.gender == 'Female'
    ).encode(
        y=alt.Y('age:O', axis=None),
        x=alt.X('sum(people):Q',
                title='population',
                sort=alt.SortOrder('descending')),
        color=alt.Color('gender:N', scale=color_scale, legend=None)
    ).mark_bar().properties(title='Female')

    middle = base.encode(
        y=alt.Y('age:O', axis=None),
        text=alt.Text('age:Q'),
    ).mark_text().properties(width=20)

    right = base.transform_filter(
        alt.datum.gender == 'Male'
    ).encode(
        y=alt.Y('age:O', axis=None),
        x=alt.X('sum(people):Q', title='population'),
        color=alt.Color('gender:N', scale=color_scale, legend=None)
    ).mark_bar().properties(title='Male')

    st.write(source)
    chart2 = alt.concat(left, middle, right, spacing=5)
    st.altair_chart(chart2)

    ###############分类柱状图#########

    pink_blue = alt.Scale(domain=('Male', 'Female'),
                      range=["steelblue", "salmon"])

    slider = alt.binding_range(min=1900, max=2000, step=10)
    select_year = alt.selection_single(name="year", fields=['year'],
                                       bind=slider, init={'year': 2000})

    chart2 = alt.Chart(source).mark_bar().encode(
        x=alt.X('sex:N', title=None),
        y=alt.Y('people:Q', scale=alt.Scale(domain=(0, 12000000))),
        color=alt.Color('sex:N', scale=pink_blue),
        column='age:O'
    ).properties(
        width=20
    ).add_selection(
        select_year
    ).transform_calculate(
        "sex", alt.expr.if_(alt.datum.sex == 1, "Male", "Female")
    ).transform_filter(
        select_year
    ).configure_facet(
        spacing=8
    )
    st.altair_chart(chart2)


    ################多子图####################
    filename = './data/vega_datasets/population.json'
    source = pd.read_json(filename)
    st.write(source)

    chart2 = alt.Chart(source).mark_area().encode(
        x='age:O',
        y=alt.Y(
            'sum(people):Q',
            title='Population',
            axis=alt.Axis(format='~s')
        ),
        facet=alt.Facet('year:O', columns=5),
    ).properties(
        title='US Age Distribution By Year',
        width=90,
        height=80
    )
    st.altair_chart(chart2)

    ################################
    st.markdown('------')
    st.markdown('## altair 画K线图')

    filename = './data/vega_datasets/ohlc.json'
    source = pd.read_json(filename)
    st.write(source)

    open_close_color = alt.condition("datum.open <= datum.close",
                                 alt.value("#06982d"),
                                 alt.value("#ae1325"))

    base = alt.Chart(source).encode(
        alt.X('date:T',
              axis=alt.Axis(
                  format='%m/%d',
                  labelAngle=-45,
                  title='Date in 2009'
              )
        ),
        color=open_close_color
    )

    rule = base.mark_rule().encode(
        alt.Y(
            'low:Q',
            title='Price',
            scale=alt.Scale(zero=False),
        ),
        alt.Y2('high:Q')
    )

    bar = base.mark_bar().encode(
        alt.Y('open:Q'),
        alt.Y2('close:Q')
    )

    chart2 = rule + bar
    st.altair_chart(chart2)

    ##################
    st.markdown('------')
    st.markdown('## altair 画范围点图')

    filename = './data/vega_datasets/countries.json'
    source = pd.read_json(filename)
    st.write(source)

    chart = alt.layer(
        data=source
    ).transform_filter(
        filter={"field": 'country',
                "oneOf": ["China", "India", "United States", "Indonesia", "Brazil"]}
    ).transform_filter(
        filter={'field': 'year',
                "oneOf": [1955, 2000]}
    )

    chart += alt.Chart().mark_line(color='#db646f').encode(
        x='life_expect:Q',
        y='country:N',
        detail='country:N'
    )
    # Add points for life expectancy in 1955 & 2000
    chart += alt.Chart().mark_point(
        size=100,
        opacity=1,
        filled=True
    ).encode(
        x='life_expect:Q',
        y='country:N',
        color=alt.Color('year:O',
            scale=alt.Scale(
                domain=['1955', '2000'],
                range=['#e6959c', '#911a24']
            )
        )
    ).interactive()

    st.altair_chart(chart)


    ###########################
    st.markdown('------')
    st.markdown('## altair 画山脊图')
    filename = './data/vega_datasets/seattle-weather.csv'
    source = pd.read_csv(filename)
    st.write(source)

    step = 20
    overlap = 1

    chart = alt.Chart(source, height=step).transform_timeunit(
        Month='month(date)'
    ).transform_joinaggregate(
        mean_temp='mean(temp_max)', groupby=['Month']
    ).transform_bin(
        ['bin_max', 'bin_min'], 'temp_max'
    ).transform_aggregate(
        value='count()', groupby=['Month', 'mean_temp', 'bin_min', 'bin_max']
    ).transform_impute(
        impute='value', groupby=['Month', 'mean_temp'], key='bin_min', value=0
    ).mark_area(
        interpolate='monotone',
        fillOpacity=0.8,
        stroke='lightgray',
        strokeWidth=0.5
    ).encode(
        alt.X('bin_min:Q', bin='binned', title='Maximum Daily Temperature (C)'),
        alt.Y(
            'value:Q',
            scale=alt.Scale(range=[step, -step * overlap]),
            axis=None
        ),
        alt.Fill(
            'mean_temp:Q',
            legend=None,
            scale=alt.Scale(domain=[30, 5], scheme='redyellowblue')
        )
    ).facet(
        row=alt.Row(
            'Month:T',
            title=None,
            header=alt.Header(labelAngle=0, labelAlign='right', format='%B')
        )
    ).properties(
        title='Seattle Weather',
        bounds='flush'
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    ).configure_title(
        anchor='end'
    )
    st.altair_chart(chart)

    ###############
    st.markdown('------')
    st.markdown('## altair 画小提琴图')
    filename = './data/vega_datasets/cars.json'
    source = pd.read_json(filename)
    st.write(source)

    chart = alt.Chart(source).transform_density(
        'Miles_per_Gallon',
        as_=['Miles_per_Gallon', 'density'],
        extent=[5, 50],
        groupby=['Origin']
    ).mark_area(orient='horizontal').encode(
        y='Miles_per_Gallon:Q',
        color='Origin:N',
        x=alt.X(
            'density:Q',
            stack='center',
            impute=None,
            title=None,
            axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True),
        ),
        column=alt.Column(
            'Origin:N',
            header=alt.Header(
                titleOrient='bottom',
                labelOrient='bottom',
                labelPadding=0,
            ),
        )
    ).properties(
        width=100
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    )
    st.altair_chart(chart)

def alt_chart3():
    ###############
    st.markdown('------')
    st.markdown('## Bar Chart with Highlighted Segment')
    filename = './data/vega_datasets/wheat.json'
    source = pd.read_json(filename)
    st.write(source)

    #source = data.wheat()
    threshold = pd.DataFrame([{"threshold": 90}])

    bars = alt.Chart(source).mark_bar().encode(
        x="year:O",
        y="wheat:Q",
    )

    highlight = alt.Chart(source).mark_bar(color="#e45755").encode(
        x='year:O',
        y='baseline:Q',
        y2='wheat:Q'
    ).transform_filter(
        alt.datum.wheat > 90
    ).transform_calculate("baseline", "90")

    rule = alt.Chart(threshold).mark_rule().encode(
        y='threshold:Q'
    )

    chart = (bars + highlight + rule).properties(width=600)
    st.altair_chart(chart)

    ###############
    st.markdown('------')
    st.markdown('## Box Plot with Min/Max Whiskers')
    filename = './data/vega_datasets/population.json'
    source = pd.read_json(filename)
    st.write(source)

    #source = data.population.url

    chart = alt.Chart(source).mark_boxplot().encode(
        x='age:O',
        y='people:Q'
    )
    st.altair_chart(chart)

    ###############
    st.markdown('------')
    st.markdown('## Facetted Scatterplot with marginal histograms')
    filename = './data/vega_datasets/iris.json'
    source = pd.read_json(filename)
    st.write(source)

    #source = data.iris()

    base = alt.Chart(source)

    xscale = alt.Scale(domain=(4.0, 8.0))
    yscale = alt.Scale(domain=(1.9, 4.55))

    area_args = {'opacity': .3, 'interpolate': 'step'}

    points = base.mark_circle().encode(
        alt.X('sepalLength', scale=xscale),
        alt.Y('sepalWidth', scale=yscale),
        color='species',
    )

    top_hist = base.mark_area(**area_args).encode(
        alt.X('sepalLength:Q',
              # when using bins, the axis scale is set through
              # the bin extent, so we do not specify the scale here
              # (which would be ignored anyway)
              bin=alt.Bin(maxbins=20, extent=xscale.domain),
              stack=None,
              title=''
             ),
        alt.Y('count()', stack=None, title=''),
        alt.Color('species:N'),
    ).properties(height=60)

    right_hist = base.mark_area(**area_args).encode(
        alt.Y('sepalWidth:Q',
              bin=alt.Bin(maxbins=20, extent=yscale.domain),
              stack=None,
              title='',
             ),
        alt.X('count()', stack=None, title=''),
        alt.Color('species:N'),
    ).properties(width=60)

    chart = top_hist & (points | right_hist)
    st.altair_chart(chart)


    ###############
    st.markdown('------')
    st.markdown('## Normalized Parallel Coordinates Example')

    chart = alt.Chart(source).transform_window(
        index='count()'
    ).transform_fold(
        ['petalLength', 'petalWidth', 'sepalLength', 'sepalWidth']
    ).transform_joinaggregate(
         min='min(value)',
         max='max(value)',
         groupby=['key']
    ).transform_calculate(
        minmax_value=(datum.value-datum.min)/(datum.max-datum.min),
        mid=(datum.min+datum.max)/2
    ).mark_line().encode(
        x='key:N',
        y='minmax_value:Q',
        color='species:N',
        detail='index:N',
        opacity=alt.value(0.5)
    ).properties(width=500)
    st.altair_chart(chart)

    ###############
    st.markdown('------')
    st.markdown('## Parallel Coordinates Example')
    chart = alt.Chart(source).transform_window(
        index='count()'
    ).transform_fold(
        ['petalLength', 'petalWidth', 'sepalLength', 'sepalWidth']
    ).mark_line().encode(
        x='key:N',
        y='value:Q',
        color='species:N',
        detail='index:N',
        opacity=alt.value(0.5)
    ).properties(width=500)
    st.altair_chart(chart)

    ###############
    st.markdown('------')
    st.markdown('## Atmospheric CO2 Concentration')
    #source = data.co2_concentration.url

    source = pd.read_csv('./data/vega_datasets/co2_concentration.csv')
    #df.to_csv('./data/vega_datasets/co2_concentration.csv',index=False)
    st.write(source)

    base = alt.Chart(
        source,
        title="Carbon Dioxide in the Atmosphere"
    ).transform_calculate(
        year="year(datum.Date)"
    ).transform_calculate(
        decade="floor(datum.year / 10)"
    ).transform_calculate(
        scaled_date="(datum.year % 10) + (month(datum.Date)/12)"
    ).transform_window(
        first_date='first_value(scaled_date)',
        last_date='last_value(scaled_date)',
        sort=[{"field": "scaled_date", "order": "ascending"}],
        groupby=['decade'],
        frame=[None, None]
    ).transform_calculate(
      end="datum.first_date === datum.scaled_date ? 'first' : datum.last_date === datum.scaled_date ? 'last' : null"
    ).encode(
        x=alt.X(
            "scaled_date:Q",
            axis=alt.Axis(title="Year into Decade", tickCount=11)
        ),
        y=alt.Y(
            "CO2:Q",
            title="CO2 concentration in ppm",
            scale=alt.Scale(zero=False)
        )
    )

    line = base.mark_line().encode(
        color=alt.Color(
            "decade:O",
            scale=alt.Scale(scheme="magma"),
            legend=None
        )
    )

    text = base.encode(text="year:N")

    start_year = text.transform_filter(
      alt.datum.end == 'first'
    ).mark_text(baseline="top")

    end_year = text.transform_filter(
      alt.datum.end == 'last'
    ).mark_text(baseline="bottom")

    chart = (line + start_year + end_year).configure_text(
        align="left",
        dx=1,
        dy=3
    ).properties(width=600, height=375)

    st.altair_chart(chart)

def main(): #主程序
    #df = pd.read_excel('./data/手机银行重要业务指标_宁夏分行数据汇总表.xlsx')

    analysis = st.sidebar.selectbox('选择一个任务',['数据分析和可视化','图像分类','Altair 简单图形','Altair 画各种图形','Altair 画各种复杂图形','Altair 画各种交互图'])

    st.set_option('deprecation.showfileUploaderEncoding', False)
    if analysis=='数据分析和可视化':
        st.title('数据分析和可视化')

        # image = '1.jpg'
        # st.image(image, caption='数据源: http://http://nx.eip.ccb.com/nx/index/index.shtml/',
        #          use_column_width=True)

        # st.header('手机银行重要业务指标分析')

        # st.markdown('数据样例 :sunglasses:')

        # #df_sample = df.head(20)
        # st.write(df)

        # st.subheader('重要数据分析：单数据分析')
        # # 选择一个重要参数
        # type = st.selectbox('选择一个参数', options=df.columns[3:])
        # st.line_chart(df[type])

        # st.subheader('重要数据分析：多个数据对比分析')
        # #选择二个重要参数
        # options = st.multiselect(
        #     '重要数据分析：多个数据对比分析',
        #     df.columns[3:],
        #     [df.columns[3], df.columns[9]]) # 默认选择期末客户数、当年活跃客户数
        # #st.write('所选重要参数是:', options)
        # st.line_chart(df[options])
        # st.bar_chart(df[options])


        st.markdown('## alt 最简单绘图')
        data = pd.DataFrame({'类型': ['A', 'B', 'C', 'D'], '差额': [1, 2, 1, 2]})
        basic_chart1 = alt.Chart(data).mark_bar().encode(
            x='类型',
            y='差额',
        )
        st.altair_chart(basic_chart1)

        st.markdown('## alt 快速绘图')
        alt_iris() ##快速绘图

        st.markdown('## alt 绘制更复杂的图形')
        alt_cars() #更复杂的图形

        st.markdown('## alt 构建复合图形')
        alt_complex() #构建复合图形

    elif analysis=='Altair 简单图形':
        st.markdown('# Altair 简单图形')
        st.markdown('---')
        alt_sigle()
    elif analysis=='Altair 画各种图形':
        st.markdown('# altair 画各种图形')
        st.markdown('---')
        alt_chart1()
    elif analysis=='Altair 画各种复杂图形':
        st.markdown('# altair 画各种复杂图形')
        st.markdown('---')
        alt_chart2()
    elif analysis=='Altair 画各种交互图':
        st.markdown('# Altair 画各种交互图')
        st.markdown('---')
        alt_chart3()
    else:
        st.title('MNIST 图像分类')
        # st.write('This is to showcase how quickly image classification apps can be built ')
        # st.header('Identifying digits from Images')
        st.subheader('请上传一个图像文件：')
        file_uploader = st.file_uploader("选择一个图像文件进行分类", type="png")
        print(file_uploader)
        #model = keras.models.load_model('mnist_model', compile=False) #keras无法加载，无法执行相关机器学习算法
        if file_uploader:
            image = Image.open(file_uploader)
            st.image(image, caption='选择图像')
        if st.button('预测'):
            #add warning for image not selected
            image = np.asarray(image)

            #pred = model.predict(image.reshape(1,28,28,1)) #keras无法加载，无法执行相关机器学习算法



            import time
            my_bar = st.progress(0)
            with st.spinner('正在预测'):
                time.sleep(2)

            #st.write(pred) #keras无法加载，无法执行相关机器学习算法
            
if __name__ == '__main__':
    main()