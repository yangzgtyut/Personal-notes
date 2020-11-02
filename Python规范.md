# Python规范

## 函数式编程

### 装饰器

**装饰器本质上是一个Python函数，它可以让其他函数在不需要做任何代码变动的前提下增加额外功能，装饰器的返回值也是一个函数对象。**它经常用于有切面需求的场景，比如：插入日志、性能测试、事务处理、缓存、权限校验等场景。装饰器是解决这类问题的绝佳设计，有了装饰器，我们就可以抽离出大量与函数功能本身无关的雷同代码并继续重用。概括的讲，装饰器的作用就是为已经存在的对象添加额外的功能。

先来看一个简单例子：

```python
def foo():
    print('i am foo')
```

现在有一个新的需求，希望可以记录下函数的执行日志，于是在代码中添加日志代码：

```python
def foo():
    print('i am foo')
    logging.info("foo is running")
```

bar()、bar2()也有类似的需求，怎么做？再写一个logging在bar函数里？这样就造成大量雷同的代码，为了减少重复写代码，我们可以这样做，重新定义一个函数：专门处理日志 ，日志处理完之后再执行真正的业务代码

```python
def use_logging(func):
    logging.warn("%s is running" % func.__name__)
    func()

def bar():
    print('i am bar')

use_logging(bar)
```

**逻辑上不难理解， 但是这样的话，我们每次都要将一个函数作为参数传递给use_logging函数。而且这种方式已经破坏了原有的代码逻辑结构，之前执行业务逻辑时，执行运行bar()，但是现在不得不改成use_logging(bar)。**那么有没有更好的方式的呢？当然有，答案就是装饰器。

#### 简单装饰器

```python
def use_logging(func):

    def wrapper(*args, **kwargs):
        logging.warn("%s is running" % func.__name__)
        return func(*args, **kwargs)
    return wrapper

def bar():
    print('i am bar')

bar = use_logging(bar)
bar()
```

函数`use_logging`就是装饰器，它把执行真正业务方法的`func`包裹在函数里面，看起来像`bar`被`use_logging`装饰了。在这个例子中，函数进入和退出时 ，被称为一个横切面(Aspect)，这种编程方式被称为面向切面的编程(Aspect-Oriented Programming)。

`@`符号是装饰器的语法糖，在定义函数的时候使用，避免再一次赋值操作：

```python
def use_logging(func):

    def wrapper(*args, **kwargs):
        logging.warn("%s is running" % func.__name__)
        return func(*args)
    return wrapper

@use_logging
def foo():
    print("i am foo")

@use_logging
def bar():
    print("i am bar")

bar()
```

如上所示，这样我们就可以省去`bar = use_logging(bar)`这一句了，直接调用`bar()`即可得到想要的结果。**如果我们有其他的类似函数，我们可以继续调用装饰器来修饰函数，而不用重复修改函数或者增加新的封装。**这样，我们就提高了程序的可重复利用性，并增加了程序的可读性。 

装饰器在Python使用如此方便都要归因于Python的函数能像普通的对象一样能作为参数传递给其他函数，可以被赋值给其他变量，可以作为返回值，可以被定义在另外一个函数内。

#### 带参数的装饰器

装饰器还有更大的灵活性，例如带参数的装饰器：在上面的装饰器调用中，比如`@use_logging`，该装饰器唯一的参数就是执行业务的函数。**装饰器的语法允许我们在调用时，提供其它参数，比如`@decorator(a)`。这样，就为装饰器的编写和使用提供了更大的灵活性。**

```python
def use_logging(level):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if level == "warn":
                logging.warn("%s is running" % func.__name__)
            return func(*args)
        return wrapper

    return decorator

@use_logging(level="warn")
def foo(name='foo'):
    print("i am %s" % name)

foo()
```

**上面的use_logging是允许带参数的装饰器。它实际上是对原有装饰器的一个函数封装，并返回一个装饰器。**我们可以将它理解为一个含有参数的闭包。当我 们使用`@use_logging(level="warn")`调用的时候，Python能够发现这一层的封装，并把参数传递到装饰器的环境中。

#### 类装饰器

再来看看类装饰器，相比函数装饰器，类装饰器具有灵活度大、高内聚、封装性等优点。使用类装饰器还可以依靠类内部的`__call__`方法，当使用 `@` 形式将装饰器附加到函数上时，就会调用此方法。

```python
class Foo(object):
    def __init__(self, func):
    	self._func = func

	def __call__(self):
        print ('class decorator runing')
    	self._func()
    	print ('class decorator ending')

@Foo
def bar():
    print ('bar')

bar()
```

#### functools.wraps

使用装饰器极大地复用了代码，但是他有一个缺点就是原函数的元信息不见了，比如函数的`__docstring__`、`__name__`、参数列表，先看例子：

装饰器

```python
def logged(func):
    def with_logging(*args, **kwargs):
        print func.__name__ + " was called"
        return func(*args, **kwargs)
    return with_logging
```

函数

```python
@logged
def f(x):
   """does some math"""
   return x + x * x
```

该函数完成等价于：

```python
def f(x):
    """does some math"""
    return x + x * x
f = logged(f)
```

不难发现，函数`f`被`with_logging`取代了，当然它的`__docstring__`，`__name__`就是变成了`with_logging`函数的信息了。

```python
print f.__name__    # prints 'with_logging'
print f.__doc__     # prints None
```

这个问题就比较严重的，好在我们有f`unctools.wraps`。`wraps`本身也是一个装饰器，它能把原函数的元信息拷贝到装饰器函数中，这使得装饰器函数也有和原函数一样的元信息了。

```python
from functools import wraps
def logged(func):
    @wraps(func)
    def with_logging(*args, **kwargs):
        print func.__name__ + " was called"
        return func(*args, **kwargs)
    return with_logging

@logged
def f(x):
    """does some math"""
    return x + x * x

print f.__name__  # prints 'f'
print f.__doc__   # prints 'does some math'
```

#### 内置装饰器

@staticmathod、@classmethod、@property

装饰器的顺序

```python
@a
@b
@c
def f ():
```

等效于

```python
f = a(b(c(f)))
```

## 模块

请注意，每一个包目录下面都会有一个`__init__.py`的文件，这个文件是必须存在的，否则，Python就把这个目录当成普通目录，而不是一个包。`__init__.py`可以是空文件，也可以有Python代码，因为`__init__.py`本身就是一个模块，而它的模块名就是`mycompany`。

当我们试图加载一个模块时，Python会在指定的路径下搜索对应的.py文件。默认情况下，Python解释器会搜索当前目录、所有已安装的内置模块和第三方模块，搜索路径存放在`sys`模块的`path`变量中

### Python模块标准文件模板：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'Michael Liao'

import sys

def test():
    args = sys.argv
    if len(args)==1:
        print('Hello, world!')
    elif len(args)==2:
        print('Hello, %s!' % args[1])
    else:
        print('Too many arguments!')

if __name__=='__main__':
    test()
```

**第1行和第2行是标准注释，第1行注释可以让这个`hello.py`文件直接在Unix/Linux/Mac上运行，第2行注释表示.py文件本身使用标准UTF-8编码；**

**==第4行是一个字符串，表示模块的文档注释，任何模块代码的第一个字符串都被视为模块的文档注释；==**

**第6行使用`__author__`变量把作者写进去，这样当你公开源代码后别人就可以瞻仰你的大名；**

## import导入

**==相对导入与绝对导入仅用于包内部==**，即包含`__init__`的文件夹内部

假如有两个模块 `a.py` 和 `b.py` 放在同一个目录下，但是没有`__init__`，那么每一个 python 文件都是一个独立的、可以直接被其他模块导入的模块，就像你导入标准库一样，它们不存在相对导入和绝对导入的问题。**相对导入与绝对导入仅用于包内部。**

### 绝对导入

- 格式：`import A.B` 或 `from A import B`

如果是绝对导入，一个模块只能导入自身的子模块或和它的顶层模块同级别的模块及其子模块。

**==对于运行入口文件，尽量使用绝对导入：导入`sys.path`中的包和运行文件所在目录下的包。==**

实际上含有绝对导入的模块也不能被直接运行，会出现 `ImportError`：

```python
ImportError: No module named XXX
```

### 相对导入

==相对导入，一个**模块必须有包结构**且**只能导入它的顶层模块内部的模块**。==**

- 相对导入格式为 `from . import B` 或 `from ..A import B`，`.`代表当前模块，`..`代表上层模块，`...`代表上上层模块，依次类推

**==导入自定义的文件，如果是非运行入口文件（最初运行的.py文件），则需要相对导入。==**

- `from . import module_name`。导入和自己同目录下的模块。
- `from .package_name import module_name`。导入和自己同目录的包的模块。
- `from .. import module_name`。导入上级目录的模块。
- `from ..package_name import module_name`。导入位于上级目录下的包的模块。
- 当然还可以有更多的`.`，每多一个点就多往上一层目录。

相对导入可以避免硬编码带来的维护问题，例如我们改了某一顶层包的名，那么其子包所有的导入就都不能用了。**==但是存在相对导入语句的模块，不能直接运行，否则会有异常==**：

```python
ValueError: Attempted relative import in non-package
```

> 所以，**如果一个模块被直接运行，则它自己为顶层模块，不存在层次结构，所以找不到其他的相对路径，**所以如果直接运行python xx.py ，而xx.py有相对导入就会报错



## 面向对象编程

三大特点：封装、继承和多态。

数据封装：方法就是与实例绑定的函数，和普通函数不同，方法可以直接访问实例的数据；

### 实例属性和类属性

#### 实例属性

1. **==和静态语言不同，Python允许对实例变量绑定任何数据。==**
2. `self.__init__`方法定义的属性

#### 类属性

如果`Student`类本身需要绑定一个属性呢？可以直接在class中定义属性，这种属性是类属性，归`Student`类所有。

当我们定义了一个类属性后，这个属性虽然归类所有，但类的所有实例都可以访问到。

> 在编写程序的时候，千万不要对实例属性和类属性使用相同的名字，因为相同名称的实例属性将屏蔽掉类属性。

### 继承

子类获得了父类的全部功能。

**当子类和父类都存在相同的`run()`方法时，我们说，子类的`run()`覆盖了父类的`run()`，在代码运行的时候，总是会调用子类的`run()`。**这样，我们就获得了继承的另一个好处：==多态==。

**==多态真正的威力：调用方只管调用，不管细节，而当我们新增一种`Animal`的子类时，只要确保父类和子类`run()`方法编写正确，不用管原来的代码是如何调用的。这就是著名的“开闭”原则：==**

- 对扩展开放：允许新增`Animal`子类；
- 对修改封闭：不需要修改依赖`Animal`类型的`run_twice()`等函数。

> ==Python的“file-like object“就是一种鸭子类型。对真正的文件对象，它有一个`read()`方法，返回其内容。但是，许多对象，只要有`read()`方法，都被视为“file-like object“。==许多函数接收的参数就是“file-like object“，你不一定要传入真正的文件对象，完全可以传入任何实现了`read()`方法的对象。

在继承关系中，如果一个实例的数据类型是某个子类，那它的数据类型也可以被看做是父类。但是，反过来就不行。

### 访问限制：私有

#### 两个下划线：`__`

属性的名称前加上两个下划线`__`。在Python中，实例的变量名如果以`__`开头，就变成了一个==私有变量（private）==，只有内部可以访问，外部不能访问。

这样就确保了外部代码不能随意修改对象内部的状态，这样通过访问限制的保护，代码更加健壮。

> **不能直接访问`__name`是因为Python解释器对外把`__name`变量改成了`_Student__name`，所以，仍然可以通过`_Student__name`来访问`__name`变量。**
>
> 此时，若修改`__name`，例如`bart.__name = 'New Name'`，这只不过是给`bart`设置了一个`__name`属性。并没有改变原来的`__name`属性。

**但是如果外部代码要获取name和score怎么办？可以给Student类增加`get_name`和`get_score`这样的方法。**

**如果又要允许外部代码修改score怎么办？可以再给Student类增加`set_score`方法。**

> ==这样在修改的时候，可以对参数做检查，避免传入无效的参数==

#### 一个下划线：`_`

**以一个下划线开头的实例变量名，比如`_name`，这样的实例变量外部是可以访问的。**但是按照惯例，视为私有变量，最好不要随意访问。

### 判断对象类型

#### 1、`type()`

`type()`返回对应的Class类型.

#### 2、`isinstance()`

**`isinstance()`判断的是一个对象是否是该类型本身，或者位于该类型的父继承链上**。

还可以判断一个变量是否是某些类型中的一种，比如下面的代码就可以判断是否是list或者tuple：

```
>>> isinstance([1, 2, 3], (list, tuple))
True
>>> isinstance((1, 2, 3), (list, tuple))
True
```

#### 3、`dir()`

要获得一个对象的所有属性和方法，可以使用`dir()`函数，它返回一个包含字符串的list。

配合`getattr()`、`setattr()`以及`hasattr()`，我们可以直接检查和操作一个对象的状态（属性和方法）。

## Python中的特殊变量和特殊方法

### 特殊变量：

特殊变量不是私有的

1. `__name__`

   函数对象有一个`__name__`属性，代表函数的名字

   由于Python中没有`main()`函数，因此当将运行python程序的命令提供给解释器时，将执行0级缩进的代码。但是，在此之前，它将定义一些特殊变量。`__name__`是这样的特殊变量之一。如果源文件作为主程序执行，则解释器将`__name__`变量设置为具有值“ `__main__`”。如果从另一个模块导入该文件，则将`__name__`设置为模块的名称。

2. `__author__`

3. `__doc__`：文档注释

4. `__all__`：可用于模块导入时限制

   被导入模块若定义了__all__属性，则只有__all__内指定的属性、方法、类可被导入。
   若没定义，则导入模块内的所有公有属性，方法和类 。

5. `__file__`

   ```python
   import os
   print(os.__file__)
   ```

6. `__slots__`

   Python允许在定义class的时候，定义一个特殊的`__slots__`变量，来限制该class实例能添加的属性。

   `__slots__`定义的属性仅对当前类实例起作用，对继承的子类是不起作用的

   **除非在在子类中也定义`__slots__`，这样，子类实例允许定义的属性就是自身的`__slots__`加上父类的`__slots__`。**

7. `__metaclass__`

### 特殊方法

### `__len__()`

类似`__xxx__`的属性和方法在Python中都是有特殊用途的，比如`__len__`方法返回长度。**==在Python中，如果你调用`len()`函数试图获取一个对象的长度，实际上，在`len()`函数内部，它自动去调用该对象的`__len__()`方法。==**所以，下面的代码是等价的：

```python
>>> len('ABC')
3
>>> 'ABC'.__len__()
3
```

**==我们自己写的类，如果也想用`len(myObj)`的话，就自己写一个`__len__()`方法：==**

```python
>>> class MyDog(object):
...     def __len__(self):
...         return 100
...
>>> dog = MyDog()
>>> len(dog)
100
```

剩下的都是普通属性或方法，比如`lower()`返回小写的字符串：

```python
>>> 'ABC'.lower()
'abc'
```

### `__str__()`

### `__repr__()`

### `__iter__()`和`__next__()`

- `__iter__`返回自身
- `__next__`进行迭代
- `__next__`（可能需要）定义循环停止条件

### `__getitem__()`

### `__setitem__()`

### `__delitem__()`

### `__getattr__()`

### `__call__()`

### with上下文管理器：`__enter__()`和`__exit()__`

with所求值的对象必须有一个`__enter__()`方法，一个`__exit__()`方法。

紧跟with后面的语句被求值后，返回对象的`__enter__()`方法被调用，这个方法的返回值将被赋值给as后面的变量。当with后面的代码块全部被执行完之后，将调用前面返回对象的`__exit__()`方法。

## 迭代器、可迭代对象、生成器

> （1）什么是可迭代对象？ 可迭代对象要么实现了能返回迭代器的 **`__iter__()`** 方法，要么实现了 **`__getitem__()`** 方法而且其参数是从零开始的索引。
>
> （2）什么是迭代器？ 迭代器是这样的对象：实现了无参数的 **`__next__()`** 方法，返回下一个元素，如果没有元素了，那么抛出 StopIteration 异常；并且实现**`__iter__()`** 方法，返回迭代器本身。
>
> （3）什么是生成器？ 生成器是带有 yield 关键字的函数。调用生成器函数时，会返回一个生成器对象。
>
> （4）什么是生成器表达式？ 生成器表达式是创建生成器的简洁句法，这样无需先定义函数再调用。

**可迭代的对象**： 使用 `__iter__` 内置函数可以获取迭代器的对象。即要么对象实现了能返回迭代器的 `__iter__` 方法，要么对象实现了 `__getitem__` 方法，而且其参数是从零开始的索引。

> 序列可以迭代的原因： `__iter__` 函数。解释器需要迭代对象 x 时，会自动调用 `__iter__(x)`。内置的  `__iter__`  函数有以下作用：
>
> (1) 检查对象是否实现了  `__iter__`  方法，如果实现了就调用它，获取一个迭代器。
>
> (2) 如果没有实现  `__iter__`  方法，但是实现了 **`__getitem__`** 方法，而且其参数是从零开始的索引，Python 会创建一个迭代器，尝试按顺序（从索引 0 开始）获取元素。
>
> (3) 如果前面两步都失败，Python 抛出 TypeError 异常，通常会提示“C objectis not iterable”（C 对象不可迭代），其中 C 是目标对象所属的类。

### 1、使用`getitem` 方法

**推荐**

```python
class Eg1:
	def __init__(self, text):
		self.text = text
		self.sub_text = text.split(' ')
	def __getitem__(self, index):
		return self.sub_text[index]
		
o1 = Eg1('Hello, the wonderful new world!')
for i in o1:
	print(i)
```

### 2、迭代器

标准的迭代器接口有两个方法：

1. `__next__`

   返回下一个可用的元素，如果没有元素了，抛出 StopIteration异常。

2. `__iter__`

   返回 `self`，以便在应该使用可迭代对象的地方使用迭代器，例如在 for 循环中。

```python
class Eg2:
	def __init__(self, text):
		self.text = text
		self.sub_text = text.split(' ')
	def __iter__(self):
		return Eg2Iterator(self.sub_text)
classEg2Iterator:
	def __init__(self, sub_text):
		self.sub_text = sub_text
		self.index = 0
	def __next__(self):
		try:
			subtext = self.sub_text[self.index]
		except IndexError:
			raise StopIteration()
		self.index += 1
		return subtext
	def __iter__(self):
		return self
```

> 我们创建了`Eg2`类，并为它实现了 `iter` 方法，此方法返回一个迭代器`Eg2Iterator`。 `Eg2Iterator` 实现了我们之前所说的`next`和`iter`方法。 

### 可迭代的对象和迭代器之间的关系：

**==Python 从可迭代的对象中获取迭代器。==**

`iter`方法从我们自己创建的迭代器类中获取迭代器，而`getitem`方法是python内部自动创建迭代器。

### 生成器

更符合 Python 习惯的方式实现 `Eg2`类。

```python
class Eg3:
	def __init__(self, text):
		self.text = text
		self.sub_text = text.split(' ')
	def __iter__(self):
		for item inself.sub_text:
			yield item
```

> 这里我们使用了yield 关键字， 只要 Python 函数的定义体中有 yield 关键字，该函数就是生成器函数。调用生成器函数时，会返回一个生成器对象。也就是说，生成器函数是生成器工厂。

上述代码还可以使用yield from进一步简化：

```python
class Eg4:
	def __init__(self, text):
		self.text = text
		self.sub_text = text.split(' ')
	def __iter__(self):
		yieldfromself.sub_text

o4 = Eg4('Hello, the wonderful new world!')
for i in o4:
	print(i)
    
'''
Hello,
the
wonderful
new
world!
'''
```

#### 生成器表达式

使用生成器表达式例子4的代码可以修改为：

```python
class Eg5:
	def __init__(self, text):
		self.text = text
		self.sub_text = text.split(' ')
	def __iter__(self):
		return (item for item inself.sub_text)

```

**==在python中，所有生成器都是迭代器。==**

## Python命名规范|模块,类名,函数,变量名,常量

```
module_name, package_name, ClassName, method_name, ExceptionName, function_name, GLOBAL_VAR_NAME, instance_var_name, function_parameter_name, local_var_name.
```

#### 1、模块

**模块尽量使用小写命名，首字母保持小写，尽量不要用下划线(除非多个单词，且数量不多的情况)**

```python
# 正确的模块名
import decoder
import html_parser

# 不推荐的模块名
import Decoder
```

#### 2、类名

**类名使用驼峰(CamelCase)命名风格，首字母大写，私有类可用一个下划线开头**

```python
class Farm():
    pass

class AnimalFarm(Farm):
    pass

class _PrivateFarm(Farm):
    pass
```

- 将相关的类和顶级函数放在同一个模块里. 不像Java, 没必要限制一个类一个模块。

#### 3、函数

**函数名一律小写，如有多个单词，用下划线隔开**

```python
def run():
    pass

def run_with_env():
    pass
```

- ==私有函数在函数前加一个下划线_==

```python
class Person():

    def _private_func():
        pass
```

#### 4、变量名

**变量名尽量小写, 如有多个单词，用下划线隔开**

```python
if __name__ == '__main__':
    count = 0
    school_name = ''
```

#### 5、常量

- 常量使用以下划线分隔的大写命名

```python
MAX_CLIENT = 100
MAX_CONNECTION = 1000
CONNECTION_TIMEOUT = 600
```