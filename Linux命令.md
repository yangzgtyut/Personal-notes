# ——Linux命令

## 统计文件夹下的文件数目

Linux下有三个命令：`ls`、`grep`、`wc`。通过这三个命令的组合可以统计目录下文件及文件夹的个数。

- 统计当前目录下文件的个数（不包括目录）

```bash
$ ls -l | grep "^-" | wc -l
```

- 统计当前目录下文件的个数（包括子目录）

```bash
$ ls -lR| grep "^-" | wc -l
```

- 查看某目录下文件夹(目录)的个数（包括子目录）

```bash
$ ls -lR | grep "^d" | wc -l
```

**命令解析：**

- `ls -l`

  长列表输出该目录下文件信息(注意这里的文件是指目录、链接、设备文件等)，每一行对应一个文件或目录，`ls -lR`是列出所有文件，包括子目录。

- `grep "^-"`

  过滤`ls`的输出信息，只保留一般文件，只保留目录是`grep "^d"`。

- `wc -l`

  统计输出信息的行数，统计结果就是输出信息的行数，一行信息对应一个文件，所以就是文件的个数。

## dpkg——安装debian软件包

安装：`dpkg -i <.deb file name>`

## 解压缩

解压：`unzip`

`unzip <.zip> -d path`——解压要指定目录

压缩：`zip`

## rm——删除文件

rm -r dir——删除文件夹及其下属文件

rm -rf dir——删除文件夹及其下属文件并不过问

## 移动文件或文件夹

`mv /a /b`——将a文件移动到b

移动目录下所有文件——`*`

`mv a b`——重命名a为b

## 复制——`cp`

同`mv`

## `mkdir`——创建文件夹

## `ls`——列出当前目录下的文件及文件夹

`ls -ah`查看隐藏文件

## `cat`——查看文本

查看cuda版本：`cat /usr/local/cuda/version.txt`

## `vi`、`vim`创建、编辑文本

i：编辑
esc：退出编辑
接下来的命令需要先输入":"
w：保存
q：退出

### `ps`：查看所有进程

kill <PID>

## `screen`用法

创建会话：`screen -S <session name>`

暂离当前会话：ctrl+a+d

恢复创建的会话：`screen -r <session name>`

> 有时在恢复screen时会出现`There is no screen to be resumed matching ****`
>
> 输入命令 `screen -d ****`
>
> 然后再使用恢复命令恢复就ok了

查看已经创建的会话：`screen -ls`

退出当前会话：`exit`

# conda环境管理

## 创建虚拟环境

conda create --name <env name> python=3.8

## 删除虚拟环境

conda remove -n <env name> --all

## 查看环境

conda info -e

