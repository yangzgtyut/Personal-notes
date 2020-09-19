# DQL

# 进阶1：基础查询
/*
语法：
select [查询列表] from [表名]
类似于：System.out.println('')

特点：
1、查询表：字段/常量值/表达式/函数
2、查询出来的的东西是一个虚拟的表格
*/

USE myemployees;

# 查询单个字段
SELECT last_name FROM employees;

# 查询多个字段，顺序、个数与与原始的表没有关系
SELECT last_name, salary, email FROM employees;

# 查询表中的所有字段
SELECT
  `employee_id`,
  `phone_number`,
  `commission_pct`,
  `first_name`
FROM
  employees;
  
SELECT * FROM employees;

# 查询常量值
SELECT 100;
SELECT 'john';  # 不区分'和"

# 查询表达式
SELECT 100%98;  # 取余

# 查询函数：调用函数并返回返回值显示
SELECT VERSION();

# 起别名
/*
1. 便于理解
2. 区分重名的字段
*/
SELECT 100%98 AS 结果
SELECT last_name AS 姓, first_name AS 名 FROM employees;
SELECT last_name 姓, first_name 名 FROM employees;

SELECT salary AS "out put" FROM employees;

# 去重

# 案例：查询员工表中涉及到的部门编号
SELECT DISTINCT department_id FROM employees;

# 加号“+”的作用
/*
java中加号+的作用：
1. 运算符
2. 连接符，只要有一个操作数作为字符串
而在MySQL中，+只有一个作用————运算符
select 100+90; 两个操作数都为数值型，则作加法运算
select '100'+90; 一个操作数为字符型，则试图将该字符型数值转换为数值型。如果转换成功，则继续做加法运算，如果转换失败，则转换为0进行加法运算
select null+90; 只要其中一方为null，则结果肯定为null
*/

# 拼接————CONCAT
# 返回结果为连接参数产生的字符串。如有任何一个参数为NULL ，则返回值为 NULL。
# 案例：查询员工名和姓并连接成一个字段，并显示为姓名
SELECT CONCAT('a', 'b', 'c') AS 结果

SELECT
  CONCAT (last_name, " ", first_name) AS 姓名
FROM
  employees;

# IFNULL(exp1, exp2)————如果为null，则转换为exp2
SELECT IFNULL(commission_pct, 0) FROM employees;
SELECT 80+NULL;


# 进阶2：条件查询
/*
语法：
SELECT
  查询列表
FROM
  表名
WHERE 
  筛选条件;
执行顺序：
1、先执行FROM
2、再执行WHERE
3、最后进行SELECT

筛选条件的分类：
	1、按照条件表达式筛选
	条件运算符：> < = != <>（不等于）
	2、按逻辑表达式查询
	逻辑表达式：
		&&  ||  !  （java）
		AND OR NOT （标准）
	3、模糊查询
		LIKE
		BETWEEN AND
		IN
		IS NULL
*/

# 1. 按照条件表达式筛选
# 案例一：查询工资大于12000的员工
SELECT * FROM employees WHERE salary>12000;

# 案例二：查询部门编号不等于90的员工名和部门编号
SELECT last_name, department_id FROM employees WHERE department_id<>90;

# 2. 按逻辑表达式查询
# 查询工资在一万到两万之间的员工名、工资以及奖金
SELECT last_name, salary, commission_pct FROM employees WHERE salary>=10000 AND salary<=20000 
# 案例而：查询部门编号不是在90到110之间的，或者工资高于15000的员工信息
SELECT * FROM employees WHERE department_id<=90 OR department_id>=110 OR salary >15000

# 3. 模糊查询
/*
1.like——像什么样的
一般和通配符搭配使用
	通配符：
	%：任意多个字符，包含0个字符
	_: 任意单个字符
	转义：/ 或 escape 可以随意指定转义字符
2. between and   ==   >= and <=
- 可以提高简洁度
- 包含边界值，即闭区间
- 不能调换前后顺序

3. in() == =
用于判断某字段的值是否属于in列表中的
	①相比于or提高简洁度
	②in列表的值类型必须统一或兼容
		兼容——可以隐式的转换
			'123'和123
	③不可以使用通配符（in相当于=而不是like）
	
4. is [not] null
	注意：
		= 或者 <> 不能判断null
			<=> 安全等于
				判断是否等于
				可以判断 null	
				可读性较差 类似于<>————不等于
		IS != =，不能用来做其他
	is null pk <=>
		is null 只能用于判断null
		<=> 能用于判断null和其他
		

*/
# 1.like
# 案例1：查询员工名中包含字符a的员工信息
SELECT * FROM employees WHERE last_name LIKE '%a%';  
# %为通配符，# 字符型必须用单引号括起来

# 案例而：查询员工名中第三个字符为n的和第五个字符为l的员工名
SELECT last_name FROM employees WHERE last_name	LIKE '__n_l%';

# 案例三：查询员工名中第二个字符为下划线的员工名
SELECT last_name FROM employees WHERE last_name LIKE '_\_%';
SELECT last_name FROM employees WHERE last_name LIKE '_$_%' ESCAPE '$';

# 2.between and
# 案例一：查询员工编号在100到120 之间的所有的员工信息
SELECT * FROM employees WHERE employee_id BETWEEN 100 AND 120;
# 等价于：
SELECT * FROM employees WHERE employee_id>=100 AND employee_id<=120;

# 3. in()
# 案例一：查询员工的工种编号是it_prog、ad_vp、ad_pres中一个的员工名和工种编号
SELECT last_name, job_id FROM employees WHERE job_id IN ('it_prog','ad_vp','ad_pres');

# 4. is null
# 案例一：没有奖金的员工名和奖金率
SELECT last_name, commission_pct FROM employees WHERE commission_pct IS NOT NULL;
SELECT last_name, commission_pct FROM employees WHERE commission_pct<=>NULL;
# 案例二：工资为12000的员工信息
SELECT * FROM employees WHERE salary <=> 12000;

SELECT * FROM employees;
SELECT * FROM employees WHERE commission_pct LIKE '%%' AND last_name LIKE '%%';


# 进阶3：排序查询

# 案例：查询员工信息，要求工资从高到低排序
SELECT * FROM employees ORDER BY salary DESC;
SELECT * FROM employees ORDER BY salary ASC;

# 案例二：查询 部门编号大于等于90的，要求按照入职时间进行排
SELECT * FROM employees WHERE department_id>=90 ORDER BY hiredate ASC;


# 案例三：查询员工信息和年薪，按照年薪高低————按表达式排序
SELECT *, salary*12*(1 + IFNULL(commission_pct, 0)) AS 年薪 FROM employees ORDER BY salary*12*(1 + IFNULL(commission_pct, 0)) DESC;

# 案例四：查询员工信息和年薪，按照（别名）高低————按别名排序
SELECT *, salary*12*(1 + IFNULL(commission_pct, 0)) AS 年薪 FROM employees ORDER BY 年薪 DESC;

# 案例五：按照姓名长度显示员工姓名和工资————按照函数排序
SELECT LENGTH(last_name) AS 字节长度, last_name, salary FROM employees ORDER BY 字节长度 DESC;

# 案例六：查询员工信息，要求先按照工资排序，再按照员工编号排序————按照多个条件字段排序
SELECT * FROM employees ORDER BY salary ASC, employee_id DESC;


# 进阶4：常见函数

# 一、. 字符函数
# 1. length()————字节长度

SELECT LENGTH('张三丰');  # 一个汉字占用三个字节（utf-8）

SHOW VARIABLES LIKE '%char%';

# 2. CONCAT——拼接字符串
SELECT CONCAT(last_name, '_', first_name) FROM employees;

# 3. upper，lower
SELECT UPPER('john');
# 案例：将姓变大写，名变小写
SELECT CONCAT(UPPER(last_name), ' ', LOWER(first_name)) AS 姓名 FROM employees;

# 4. substr/substring————截取字符
# sql索引从1开始
SELECT SUBSTR('李莫愁爱上了陆展元', 7) AS output;  # 截取从该位置开始后的字符
SELECT SUBSTR('李莫愁爱上了陆展元', 1, 1) AS output;  # 截取从指定索引处指定字符长度的字符

# 案例：将字符中首字符大写，其他字符小写，然后用_拼接
SELECT CONCAT(UPPER(SUBSTR(last_name, 1, 1)), '_', LOWER(SUBSTR(last_name, 2))) FROM employees;

# 5. instr()————用于返回起始索引，若不存在，则返回0
SELECT INSTR('杨不悔爱上了殷六侠', '殷六侠') AS output FROM employees;

# 6. trim————去首尾的字符（默认空格）
# 不能去掉中间的
SELECT LENGTH(TRIM('   张翠山   ')) AS output;
SELECT TRIM('a' FROM 'aaaaa张aa翠aa山aaaaaaa') AS output;
SELECT TRIM('aa' FROM 'aaaaa张aa翠aa山aaaaaaa') AS output;

# 7. lpad————用指定的字符实现左填充指定长度
# 若指定长度小于元字符串长度，则截断
SELECT LPAD('殷素素', 10, '*') AS output;
SELECT LPAD('殷素素', 2, '*') AS output;

# 8. rpad
SELECT RPAD('殷素素', 10, '*') AS output;

# 9. replace————替换
SELECT REPLACE('张无忌爱上了周芷若，周芷若爱上了张无忌', '周芷若', '赵敏') AS output;


# 二、数学函数
# 一般第二个参数D都是选取小数点后的位数，默认为0，-1代表小数点左边第1个位置的数值

# 1. round————四舍五入，若有第二个参数，则对该位置参数四舍五入
SELECT ROUND(1.65) AS output;
SELECT ROUND(-1.65) AS output;  # 先对绝对值四舍五入，再添加负号
SELECT ROUND(1.567, 2) AS output;  # 小数点保留两位，四舍五入

# 2. ceil————向上取整，返回>=该参数的最小整数
SELECT CEIL(1.1);
SELECT CEIL(1.0);
SELECT CEIL(-1.1);

# 3. floor————向下取整，返回小于等于该参数的最大整数
SELECT FLOOR(9.99);

# 4. truncate————截断，小数点的位数
SELECT TRUNCATE(1.5777, 1);

# 5. mod————取余
# a-a/b*b————正负和被除数的正负一致
# 若除数为0，则返回null
SELECT MOD(10, 3);
SELECT MOD(-10, 3);


# 三、日期函数

# 1. now()————返回当前系统日期+时间
SELECT NOW();

# 2. curdate————返回当前系统日期，不包含时间
SELECT CURDATE();

# 3. curtime————返回当前系统时间，不包含日期
SELECT CURTIME();

# 4. 获取指定的部分（年，月，日，时，分，秒）
# YEAR, MONTH(MONTHNAME-月名), DATE, HOUR, MINUTE, SECOND
SELECT YEAR(NOW()) AS 年;
SELECT YEAR('1995-12-26') AS 生年;

SELECT YEAR(hiredate) AS 年 FROM employees; 

# 5. str_to_date: 将日期格式的字符转换成指定格式的日期
SELECT STR_TO_DATE('1995-12-26', '%Y-%c-%d') AS 生日;
# 案例：查询入职日期为1992-4-3的员工信息
SELECT * FROM employees WHERE hiredate = '1992-4-3';
SELECT * FROM employees WHERE hiredate = STR_TO_DATE('4-3-1992', '%c-%d-%Y');

# 6. date_format————将日期按照指定的格式转换为str
SELECT DATE_FORMAT(NOW(), '%Y年%m月%d日');
# 查询有奖金的员工名和入职日期(xx月/xx日 /xx年)
SELECT last_name, DATE_FORMAT(hiredate, '%m月/%d日 /%Y年') AS 入职日期 FROM employees WHERE commission_pct IS NOT NULL;


# 四、其他函数

SELECT VERSION();
SELECT DATABASE();
SELECT USER();


# 五、流程控制函数

# 1. IF————类似于三元运算符
SELECT IF(10>5, '大', '小');

# 案例：查询是否有奖金
SELECT last_name, commission_pct, IF(commission_pct IS NOT NULL, '嘻嘻', '呵呵') FROM employees;

# 2. case
# ①————类似于switch case等值判断
/*
case 要判断的字段或表达式
when 常量1 then [[常量1（值不需要加;）]或[语句1;（语句要加;）]]
when 常量1 then [[常量2]或[语句2;]]
...
else [[常量n]或[语句n;]]
end

注意：搭配select时，当作语句，则when中只能放常量不能放语句

switch(变量或表达式):
	case 常量1: 语句1; break;
	...
	default: 语句; break;
*/

# 案例
/*
查询员工的工资，若部门号=30，则显示为工资的1.1倍
查询员工的工资，若部门号=40，则显示为工资的1.2倍
查询员工的工资，若部门号=50，则显示为工资的1.3倍
其他部门，为原工资
*/
SELECT
  salary AS 原始工资,
  department_id,
  CASE
    department_id
    WHEN 30
    THEN salary * 1.1
    WHEN 40
    THEN salary * 1.2
    WHEN 50
    THEN salary * 1.3
    ELSE salary
  END AS 新工资
FROM
  employees;

# ②————类似于多重if，但不完全一样
 /*
mysql：
case
when 条件1 then 值1[/语句1;]
when 条件2 then 值2[/语句2;]
...
else 值n[语句n]
end

注意：这种情况下，case后面没有表达式或变量

java中：
if (条件1) {
	语句1;
}else if (条件2) {
	语句2;
}
...
else {
	语句n;
}
*/
# 案例：
 /*
查询员工的工资情况，
若>20000显示A级别，
若>15000显示B级别，
若>10000显示C级别，
否则，显示D级别
*/
SELECT
  salary,
  CASE
    WHEN salary > 20000
    THEN 'A'
    WHEN salary > 15000
    THEN 'B'
    WHEN salary > 10000
    THEN 'C'
    ELSE '垃圾'
  END AS 工资级别
FROM
  employees;


# 二、分组函数
/*
功能：用作统计处理

分类：
sum、avg、max、min、count（计数，非空值的个数）

特点：
1、sum、avg一般只用于处理数值型
   max、min、count可以处理任何类型
2、所有分组函数都忽略null值
3、搭配 DISTINCT 忽略重复运算
4、count函数
   一般count(*)用于统计行数
5、和分组函数一同查询的字段要求是gourp by后的字段
*/

# 1、简单使用
SELECT SUM(salary) FROM employees;
SELECT AVG(salary) FROM employees;
SELECT MAX(salary) FROM employees;
SELECT MIN(salary) FROM employees;
SELECT COUNT(salary) FROM employees;

SELECT ROUND(AVG(salary), 2) FROM employees;

# 2、参数类型

SELECT SUM(last_name), AVG(last_name) FROM employees;
SELECT SUM(hiredate), AVG(hiredate) FROM employees;

SELECT MAX(last_name), MIN(last_name) FROM employees;
SELECT MAX(hiredate), MIN(hiredate) FROM employees;

SELECT COUNT(last_name), COUNT(commission_pct) FROM employees;

3、忽略null值
SELECT SUM(commission_pct), AVG(commission_pct), SUM(commission_pct)/35, SUM(commission_pct)/107 FROM employees;

SELECT MAX(commission_pct), MIN(commission_pct) FROM employees;
SELECT COUNT(commission_pct) FROM employees;

4、搭配distinct
SELECT SUM(DISTINCT salary), SUM(salary) FROM employees;
SELECT COUNT(DISTINCT salary), COUNT(salary) FROM employees;

5、count的详细介绍
SELECT COUNT(*) FROM employees;  
# count(*)会统计所有字段，所以不可能有一行字段全部为null，所以一般用意统计表里面的总行数
SELECT COUNT(1) FROM employees;
# 效率
# myisam存储引擎下，有一个内部计数器，所以count(*)效率最高
# innodb存储引擎下，COUNT(*)和count(1)差不多，比count(字段)高（多了一个判断：是否为null）

6、和分组函数一同查询的字段有限制
SELECT AVG(salary), employee_id FROM employees;
# 和分组函数一同查询的字段要求是gourp by后的字段

# 测试
SELECT MAX(salary), MIN(salary), AVG(salary), SUM(salary) FROM employees;
SELECT DATEDIFF(MAX(hiredate), MIN(hiredate)) FROM employees;
SELECT COUNT(*) FROM employees WHERE department_id = 90;


# 进阶5：分组查询

/*
语法
select
  字段（要求出现在group by的后面）, group_function (字段)
from
  表
[where 条件]
[group by group_by_exp]
[order by expresion]

注意：查询列表比较特殊，要求是分组函数和group by后出现的字段


*/


# 案例1：查询每个工种的最高工资
SELECT
  MAX(salary) AS 最高工资,
  job_id
FROM
  employees
GROUP BY job_id;

# 案例2：查询每个位置上的部门个数
SELECT COUNT(*), location_id FROM departments GROUP BY location_id;

# 添加分组前的筛选条件
# 案例1：查询邮箱中包含a字符的，每个部门的平均工资
SELECT AVG(salary), department_id FROM employees WHERE email LIKE '%a%' GROUP BY department_id;

# 案例2：查询有奖金的每个领导手下员工的最高工资
 SELECT
  MAX (salary),
  manager_id
FROM
  employees
WHERE commission_pct IS NOT NULL
GROUP BY manager_id;

# 添加分组后的筛选条件
# having
# 从原始表中可以得到的条件放到where中，需要从二次表中才能得到的放在having中
# 案例1：查询哪个部门的员工个数大于2
 SELECT
  COUNT (*),
  department_id
FROM
  employees
GROUP BY department_id
HAVING COUNT (*) > 2 

# 案例2：查询每个工种有奖金的员工的最高工资>12000的工种编号和最高工

SELECT
  MAX (salary),
  job_id
FROM
  employees
WHERE 
  commission_pct IS NOT NULL
GROUP BY 
  job_id
HAVING 
  MAX (salary) > 12000;

SELECT
  MAX (salary),
  job_id
FROM
  employees
WHERE MAX (salary) > 12000 # 出错：原始表中并没有max(salary)
 GROUP BY job_id
HAVING commission_pct IS NOT NULL;

# 案例3：查询领导编号>102的每个领导手下的最低工资>5000的领导编号以及最低工资
SELECT
  manager_id,
  MIN (salary)
FROM
  employees
WHERE manager_id > 102
GROUP BY manager_id
HAVING MIN (salary) > 5000;

SELECT
  manager_id,
  MIN (salary)
FROM
  employees
GROUP BY manager_id
HAVING MIN (salary) > 5000
  AND manager_id > 102;


# 按照表达式分组
# 案例1：按照员工长度分组，查询每一组的员工个数，并筛选员工个数大于5的分组

SELECT
  COUNT(*),
  LENGTH(last_name)
FROM
  employees
GROUP BY LENGTH(last_name)
HAVING LENGTH(last_name)>5;  # 错误：会对分组后的虚拟表计算LENGTH(last_name)，而分组后的虚拟表是不存在last_name列的

SELECT
  COUNT(*),
  LENGTH(last_name)
FROM
  employees
GROUP BY LENGTH(last_name)
HAVING COUNT(*) > 5;  # 而count(*)由于不存在*列，所以不会出错

# 按照多个字段分组

# 案例：查询每个部门每个工种的员工的平均工资
SELECT
  AVG(salary),
  department_id,
  job_id
FROM
  employees
GROUP BY 
  department_id, job_id;
  
  
# 添加排序
# 案例：查询每个部门（部门id不为null）每个工种的员工的平均工资，且>10000的显示，并从高到低排序
SELECT
  AVG(salary),
  department_id,
  job_id
FROM
  employees
WHERE department_id IS NOT NULL
GROUP BY 
  department_id, job_id
HAVING AVG(salary) > 12000
ORDER BY AVG(salary) DESC;

-- SELECT salary, LENGTH(salary)
-- FROM employees
-- GROUP BY LENGTH(salary)
-- HAVING LENGTH(salary)>7;

-- SELECT LENGTH(100000)>5;


USE girls;
# 进阶6：多表连接
/*
笛卡尔乘积现象：表1：m行，表2：n行，结果：mn行
原因：没有有效的连接条件
避免：添加有效的连接条件

分类：
	按年代分类：
	sql92标准（仅支持内连接）
	sql99标准【推荐】
		支持内连接+外连接（左外连接+右外连接）+交叉链接
	
	按功能分类：
		内连接
			等值连接
			非等值连接
			自连接
		外连接
			左外连接
			右外连接
			全外连接
		交叉链接诶

*/
SELECT * FROM beauty;
SELECT * FROM boys;

SELECT
  NAME,
  boyName
FROM
  boys,
  beauty
WHERE beauty.`boyfriend_id` = boys.`id`


# 一、SQL92标准
# 1. 等值连接————两个表的条件使用=连接
# 使用其中一张表匹配另一个表

# 案例1：查询女生名和对应的男神名
SELECT
  NAME,
  boyName
FROM
  boys,
  beauty
WHERE beauty.`boyfriend_id` = boys.`id`

# 案例2：查询员工名和对应的部门名
SELECT last_name, department_name FROM employees, departments
WHERE employees.department_id = departments.department_id;


# 2. 为了防止有歧义的列，一般为表起别名（相当于生成了虚拟视图）
/*
1. 提高语句见解读
2. 区分多个重名的字段就不能使用原来的表名限定

> 注意:：如果起了别名，则查询的字段
*/
# 案例：查询员工名、工种名、工种号
SELECT
  last_name,
  e.job_id,
  job_title
FROM
  employees AS e,
  jobs AS j
WHERE e.`job_id` = j.`job_id`;


# 3. 两个表的顺序可以调换（使用其中一张表匹配另一个表）


# 4. 可以加筛选
# 案例：查询有奖金的员工名，部门名
 SELECT
  last_name,
  e.`department_id`
FROM
  employees e,
  departments d
WHERE e.`department_id` = d.`department_id`
  AND e.`commission_pct` IS NOT NULL;

# 案例2：查询城市名中第二个字符为o的部门名和城市名
 SELECT
  department_name,
  city
FROM
  departments d,
  locations l
WHERE d.`location_id` = l.`location_id`
  AND l.`city` LIKE '_o%';
  
# 5、加分组

# 案例1：查询每个城市的部门个数
SELECT
  COUNT (*) 部门个数,
  city
FROM
  departments d,
  locations l
WHERE d.`location_id` = l.`location_id`
GROUP BY city;

# 案例2：查询出有奖金的每个部门的部门名的领导编号以及该部门的最低工资
 SELECT
  department_name,
  e.`manager_id`,
  MIN(salary)
FROM
  departments d,
  employees e
WHERE d.`department_id` = e.`department_id`
  AND e.`commission_pct` IS NOT NULL
GROUP BY department_name;


# 6. 加排序
# 案例：查询每个工种的工种名和员工的个数，并且按照员工个数降序
SELECT
  job_title,
  COUNT(*)
FROM
  employees e,
  jobs j
WHERE e.`job_id` = j.`job_id`
GROUP BY job_title
ORDER BY COUNT(*) DESC;


# 7. 多表连接

# 案例：查询员工名、部门名和所在的城市
SELECT
  last_name,
  department_name,
  city
FROM
  employees e,
  departments d,
  locations l
WHERE e.`department_id` = d.`department_id`
  AND d.`location_id` = l.`location_id`
ORDER BY department_name DESC;



# 2. 非等值连接
/*CREATE TABLE job_grades
(grade_level VARCHAR(3),
 lowest_sal  INT,
 highest_sal INT);

INSERT INTO job_grades
VALUES ('A', 1000, 2999);

INSERT INTO job_grades
VALUES ('B', 3000, 5999);

INSERT INTO job_grades
VALUES('C', 6000, 9999);`stuinfo`

INSERT INTO job_grades
VALUES('D', 10000, 14999);

INSERT INTO job_grades
VALUES('E', 15000, 24999);

INSERT INTO job_grades
VALUES('F', 25000, 40000);
*/

# 案例1：查询员工的工资和工资级别`job_grades`
SELECT salary, employee_id FROM employees;
SELECT
  salary,
  grade_level,
  employee_id
FROM
  employees e,
  job_grades j
WHERE salary BETWEEN j.`lowest_sal`
  AND j.`highest_sal`
ORDER BY employee_id;


# 3. 自连接

# 案例： 查询员工名以及上级的名称
SELECT e.employee_id, e.last_name, m.employee_id, m.last_name
FROM employees e, employees m
WHERE e.`manager_id`=m.`employee_id`;

# 测试
SELECT PASSWORD('杨子固');





# 二、sql99标准
# 1. 内连接
# ①等值连接
# 案例1：查询员工名和对应的部门名
SELECT
  last_name,
  department_name
FROM
  employees e
  INNER JOIN departments d
    ON e.`department_id` = d.`department_id`;

# 案例2：查询名字中包含e的员工名、工种名
SELECT
  last_name,
  job_title
FROM
  employees e
  INNER JOIN jobs j
    ON e.`job_id` = j.`job_id`
WHERE e.`last_name` LIKE '%e%';

# 案例3：查询部门个数>3的城市名和部门个数
 SELECT
  city,
  COUNT (*)
FROM
  departments d
  INNER JOIN locations l
    ON d.`location_id` = l.`location_id`
GROUP BY city
HAVING COUNT (*) > 3;

# 案例4：查询哪个部门的员工个数>3的部门名和员工个数，并按照员工个数降序排序
 SELECT
  department_name,
  COUNT (*)
FROM
  departments d
  INNER JOIN employees e
    ON d.`department_id` = e.`department_id`
GROUP BY department_name
HAVING COUNT (*) > 3
ORDER BY COUNT (*) DESC;

# 案例5：查询员工名、工种名、部门名，并按照部门名降序
 SELECT
  last_name,
  job_title,
  d.department_name
FROM
  employees e
  INNER JOIN jobs j
    ON e.`job_id` = j.`job_id`
  INNER JOIN departments d
    ON e.`department_id` = d.`department_id`
ORDER BY d.department_name DESC;


# ②非等值连接

# 案例：查询员工的工资级别
 SELECT
  last_name,
  grade_level,
  salary
FROM
  employees e
  JOIN job_grades g
    ON e.`salary` BETWEEN g.`lowest_sal`
    AND g.`highest_sal`;

# 查询每个工资级别的个数>2的，并按照工资级别降序排序
 SELECT
  COUNT(*),
  grade_level
FROM
  employees e
  JOIN job_grades g
    ON e.`salary` BETWEEN g.`lowest_sal`
    AND g.`highest_sal`
GROUP BY grade_level
HAVING COUNT(*) > 20
ORDER BY grade_level DESC;


# ③自连接
# 案例：查询员工的名字、上级的名字
 SELECT
  e.`last_name` 员工,
  m.`last_name` 上级
FROM
  employees e
  JOIN employees m
    ON e.`manager_id` = m.`employee_id`
    
    
# 2. 外连接


# 引入：查询男朋友不在表中的女神名
USE girls;
SELECT * FROM beauty;

# 左外连接
SELECT
  b.`name`
FROM
  beauty b
  LEFT OUTER JOIN boys bo
    ON b.`boyfriend_id` = bo.`id`
WHERE bo.`id` IS NULL;

# 右外连接
SELECT
  b.`name`
FROM
  boys bo
  RIGHT OUTER JOIN beauty b
    ON b.`boyfriend_id` = bo.`id`
WHERE bo.`id` IS NULL;

# 案例：查询那个部门没有员工
USE myemployees;

SELECT
  department_name
FROM
  departments d
  LEFT OUTER JOIN employees e
    ON d.`department_id` = e.`department_id`
WHERE e.`employee_id` IS NULL
GROUP BY d.`department_id`;


SELECT
  department_name
FROM
  employees e
  RIGHT OUTER JOIN departments d
    ON d.`department_id` = e.`department_id`
WHERE e.`employee_id` IS NULL
GROUP BY d.`department_id`;


USE girls;

SELECT
  b.*,
  bo.*
FROM
  beauty b 
  FULL OUTER JOIN boys bo
    ON b.`boyfriend_id` = bo.`id`;

# 3. 交叉链接
SELECT
  b.*,
  bo.*
FROM
  beauty b
  CROSS JOIN boys bo;


# 进阶7：子查询

# 1. where或having后面

# ① 标量子查询


# 案例1：谁的工资比Abel高

SELECT
  last_name,
  salary
FROM
  employees
WHERE salary >
  (SELECT
    salary
  FROM
    employees
  WHERE last_name = 'abel');

# 案例2：返回job_id与141号员工相同，salary比141员工多的员工姓名，job_id和工资
SELECT
  last_name,
  job_id,
  salary
FROM
  employees
WHERE salary >
  (SELECT
    salary
  FROM
    employees
  WHERE employee_id = 143)
  AND job_id =
  (SELECT
    job_id
  FROM
    employees
  WHERE employee_id = 141)
ORDER BY salary;

# 案例3：返回公司工资最少的员工的last_name,job_id,salary
SELECT
  last_name,
  job_id,
  salary
FROM
  employees
WHERE salary =
  (SELECT
    MIN(salary)
  FROM
    employees);

# 案例4：查询最低工资大于50号部门最低工资的部门id和其最低工资
SELECT
  MIN(salary),
  department_id
FROM
  employees
GROUP BY department_id
HAVING MIN(salary) >
  (SELECT
    MIN(salary)
  FROM
    employees
  WHERE department_id = 50);


# ②列子查询（多行子查询）
# 案例1：返回location_id是1400或1700的部门中所有员工姓名
SELECT
  last_name,
  department_id
FROM
  employees
WHERE department_id IN
  (SELECT
    department_id
  FROM
    departments
  WHERE location_id IN (1400, 1700));
# 用内连接更简单
SELECT
  last_name,
  d.department_id, 
  location_id
FROM
  employees e
  INNER JOIN departments d
    ON e.`department_id` = d.`department_id`
WHERE location_id IN (1400, 1700);

# 案例2：返回其他工种中比job_id为'IT_PROG'部门任一工资低的员工的员工号，job_id以及salary
SELECT
  employee_id,
  job_id,
  salary,
  department_id
FROM
  employees e
WHERE job_id != 'IT_PROG'
  AND salary < ANY
  (SELECT DISTINCT
    salary
  FROM
    employees
  WHERE job_id = 'IT_PROG');


# 案例3：返回其他工种中比job_id为'IT_PROG'部门任一工资低的员工的员工号，job_id以及salary

SELECT
  employee_id,
  job_id,
  salary,
  department_id
FROM
  employees e
WHERE job_id != 'IT_PROG'
  AND salary < ALL
  (SELECT DISTINCT
    salary
  FROM
    employees
  WHERE job_id = 'IT_PROG');


# ③行子查询（结果为一行对列或者多行多列）
# 多个判断条件都是=
# 案例：查询员工编号最小的并且工资最高的员工信息
SELECT
  *
FROM
  employees
WHERE (employee_id, salary) =
  (SELECT
    MIN(employee_id),
    MAX(salary)
  FROM
    employees);
    
    
SELECT
  *
FROM
  employees
WHERE salary >= ALL
  (SELECT
    salary
  FROM
    employees)
  AND employee_id <= ALL
  (SELECT
    employee_id
  FROM
    employees);
    
    
# 二、放在select后面的子查询（仅支持用标量子查询）

# 案例1：查询每个部门的员工个数，及部门信息

# ~不能用group by，因为select要查询涉及到两个表，而不是只查询一个表~
# 错误：group by无法统计员工数为null的部门;
SELECT
  d.*,
  COUNT(*) 员工个数
FROM
  employees e
  RIGHT OUTER JOIN departments d
    ON e.`department_id` = d.`department_id`
GROUP BY e.`department_id`;

   
SELECT
  d.*,
  (SELECT
    COUNT(*)
  FROM
    employees AS e
  WHERE e.`department_id` = d.`department_id`) 员工个数
FROM
  departments AS d;

#案例2：查询员工号=102的部门名
SELECT
  (SELECT
    department_name
  FROM
    departments d
    INNER JOIN employees e
      ON d.department_id = e.department_id
  WHERE e.employee_id = 102) 部门名称;

# 用连接查询
SELECT
  department_name
FROM
  departments d
  INNER JOIN employees e
    ON d.`department_id` = e.`department_id`
WHERE employee_id = 102;

# 用标量子查询
SELECT
  department_name
FROM
  departments d
WHERE department_id =
  (SELECT
    department_id
  FROM
    employees e
  WHERE e.`employee_id` = 102);
  
  
#三、from后面
# 案例：查询每个部门的平均工资的工资等级
 ①查询每个部门的平均工资
SELECT
  AVG (salary),
  department_id
FROM
  employees e
GROUP BY department_id;

②查询不同的工资等级
SELECT
  job_grades
FROM
  jobs;

③将①的虚拟表和job_grades连接
SELECT
  a.*,
  j.`grade_level`
FROM
  (SELECT
    AVG(salary) 平均工资,
    department_id
  FROM
    employees e
  GROUP BY department_id) a
  INNER JOIN job_grades j
    ON a.平均工资 BETWEEN j.`lowest_sal`
    AND j.`highest_sal`;


# 四、放在`exists`后面的子查询——相关子查询（较少）
# 先查询外查询，在查询内查询
#内查询涉及到了
# 案例1：查询有员工的部门名
SELECT
  department_name
FROM
  departments d
WHERE EXISTS
  (SELECT
    *
  FROM
    employees e
  WHERE d.`department_id` = e.`department_id`);

# 能用exists的绝对能用IN
SELECT
  department_name
FROM
  departments d
WHERE d.`department_id` IN
  (SELECT DISTINCT
    department_id
  FROM
    employees e
  GROUP BY department_id);



# 测试1
#1：查询和Zlotkey相同部门的员工姓和工资
SELECT
  last_name,
  salary
FROM
  employees
WHERE department_id =
  (SELECT
    department_id
  FROM
    employees
  WHERE last_name = 'Zlotkey')
  AND last_name != 'Zlotkey';

# 2：查询工资比公司平均工资高的员工的员工号、姓名、工资
SELECT
  employee_id,
  last_name,
  salary
FROM
  employees
WHERE salary >
  (SELECT
    AVG (salary)
  FROM
    employees);

# 3：查询各部门工资中比本部门平均工资的员工的员工号、姓名、工资
SELECT
  employee_id,
  last_name,
  salary,
  tab2.`department_id`,
  tab1.部门平均工资
FROM
  (SELECT
    AVG (salary) 部门平均工资,
    department_id d_id
  FROM
    employees
  GROUP BY department_id) tab1
  INNER JOIN employees tab2
    ON tab1.d_id = tab2.department_id
WHERE tab2.`salary` > tab1.部门平均工资;

# 4：查询和姓名中包含字母u的员工在相同部门的员工的员工号和姓名
SELECT
  employee_id,
  last_name
FROM
  employees
WHERE department_id IN
  (SELECT
    DISTINCT department_id
  FROM
    employees
  WHERE last_name LIKE '%u%');


# 5：查询在部门的location_id为1700的部门工作员工的员工号
# 使用内连接
SELECT
  employee_id,
  location_id
FROM
  employees e
  INNER JOIN departments d
    ON e.`department_id` = d.`department_id`
WHERE d.`location_id` = 1700;

# 使用子查询
SELECT
  employee_id
FROM employees
WHERE department_id IN 
(SELECT department_id
FROM departments
WHERE location_id=1700);


# 6：查询管理者是K_ing的员工姓名和工资
# 先查询K_ing员工，再查哪个员工的manager_id=K_ing的id
SELECT
  last_name,
  salary
FROM
  employees
WHERE manager_id IN
  (SELECT
    employee_id
  FROM
    employees
  WHERE last_name = 'K_ing');



# 7、查询工资最高的员工的姓名，要求first_name和last_name 显示为一列，列名为姓名
SELECT
  CONCAT(first_name, ' ', last_name) AS '姓.名'
FROM
  employees
WHERE salary =
  (SELECT
    MAX(salary)
  FROM
    employees);
    
    
# 测试2
# 1. 查询工资最低的员工信息: last_name, salary
SELECT
  last_name,
  salary
FROM
  employees
WHERE salary =
  (SELECT
    MIN (salary)
  FROM
    employees);

# 2. 查询平均工资最低的部门信息
 SELECT
  *
FROM
  departments
WHERE department_id =
  (SELECT
    department_id
  FROM
    employees
  GROUP BY department_id
  ORDER BY AVG (salary)
  LIMIT 1);

SELECT
  *
FROM
  departments
WHERE department_id =
  (SELECT
    tab1.department_id
  FROM
    (SELECT
      MIN (tab0.平均工资),
      tab0.department_id  # 错误：和分组函数一同出现的字段必须是group by后的字段
    FROM
      (SELECT
        department_id,
        AVG (salary) 平均工资
      FROM
        employees
      GROUP BY department_id) tab0) tab1);
      

# 套娃子查询：查最低平均工资 -> 查最低工资的部门id -> 查该部门的信息
SELECT
  *
FROM
  departments
WHERE department_id =
  (SELECT
    tab1.department_id
  FROM
    (
    # 查最低平均工资的编号    
    SELECT
      department_id,
      AVG(salary)
    FROM
      employees
    GROUP BY department_id
    HAVING AVG(salary) =
      (
      # 查最低平均工资      
      SELECT
        MIN(tab0.平均工资)
      FROM
        (SELECT
          department_id,
          AVG (salary) 平均工资
        FROM
          employees
        GROUP BY department_id) tab0)) tab1);


 

# 3. 查询平均工资最低的部门信息和该部门的平均工资
# 使用from表子查询
SELECT
  d.*,
  tab1.平均工资
FROM
  (SELECT
    department_id d_id,
    AVG (salary) 平均工资
  FROM
    employees
  GROUP BY department_id) tab1
  INNER JOIN departments d
    ON tab1.d_id = d.`department_id`
ORDER BY tab1.平均工资
LIMIT 1;

# 4. 查询平均工资最高的 job 信息
SELECT
  *
FROM
  jobs
WHERE jobs.`job_id` =
  (SELECT
    job_id
  FROM
    employees
  GROUP BY job_id
  ORDER BY AVG (salary) DESC
  LIMIT 1);


# 5. 查询平均工资高于公司平均工资的部门有哪些?
SELECT
  department_id,
  AVG (salary) 部门平均工资
FROM
  employees
GROUP BY department_id
HAVING 部门平均工资 >
  (SELECT
    AVG (salary)
  FROM
    employees);


SELECT
  department_id
FROM
  (SELECT
    department_id,
    AVG (salary) 部门平均工资,
    公司平均工资
  FROM
    employees
    INNER JOIN
      (SELECT
        AVG (salary) 公司平均工资
      FROM
        employees) tab1
  GROUP BY department_id) tab2
WHERE tab2.部门平均工资 > tab2.公司平均工资;

# 6. 查询出公司中所有 manager 的详细信息.
SELECT
  *
FROM
  employees
WHERE employee_id IN
  (SELECT DISTINCT
    manager_id
  FROM
    employees);

# 7. 各个部门中 最高工资中最低的那个部门 的 最低工资是多少
SELECT
  MIN(salary),
  department_id
FROM
  employees
WHERE department_id =
  (SELECT
    department_id
  FROM
    employees
  GROUP BY department_id
  ORDER BY MAX(salary)
  LIMIT 1);

# 8. 查询平均工资最高的部门的 manager 的详细信息: last_name, department_id, email, salary
SELECT
  last_name,
  department_id,
  email,
  salary
FROM
  employees e
WHERE e.`employee_id` IN
  (SELECT
    DISTINCT manager_id
  FROM
    employees
  WHERE department_id =
    (SELECT
      department_id
    FROM
      employees
    GROUP BY department_id
    ORDER BY AVG (salary) DESC
    LIMIT 1));




# 进阶8：分页查询
# 案例1：显示前5条员工信息
 SELECT
  *
FROM
  employees
LIMIT 0, 5;

# 案例2：查询第11条——25条
SELECT * FROM employees LIMIT 10, 15;

# 案例3：查询有奖金的员工的信息，并且工资较高的前10位显示出来
SELECT
  *
FROM
  employees
WHERE commission_pct IS NOT NULL
ORDER BY salary DESC
LIMIT 10;



# 进阶9：联合查询

# 引入案例：查询部门编号>90或邮箱中包含a的员工信息

SELECT * FROM employees WHERE email LIKE '%a%' OR department_id > 90;

SELECT
  *
FROM
  employees
WHERE email LIKE '%a%'
UNION
SELECT
  *
FROM
  employees
WHERE department_id > 90;


/*
test 数据库希望帮到你们
create database country;
 
use country;
 
create table t_ca(
id int not null primary key,
cname varchar(30) null,
csex varchar(10) null
);
 
insert into t_ca (id,cname,csex) values (1,'韩梅梅','女');
insert into t_ca (id,cname,csex) values (2,'李雷','男');
insert into t_ca (id,cname,csex) values (3,'李明','男');
 
create table t_ua(
t_id int not null primary key,
t_name varchar(30) null,
t_gender varchar(30) null
)
 
insert into t_ua (t_id,t_name,t_gender) values (1,'john','male');
insert into t_ua (t_id,t_name,t_gender) values (2,'lucy','female');
insert into t_ua (t_id,t_name,t_gender) values (3,'lily','female');
 
insert into t_ua (t_id,t_name,t_gender) values (4,'jack','male');
insert into t_ua (t_id,t_name,t_gender) values (5,'rose','female');

*/


# 案例：查询中国用户中男性用户信息，以及外国用户中男性用户信息

SELECT * FROM t_ca WHERE csex='男'
UNION
SELECT * FROM t_ua WHERE t_gender='male';
























# DML

# 一、插入insert

# 方式1
INSERT INTO beauty(id, NAME, sex, borndate, phone, photo, boyfriend_id)
VALUE(13, '唐艺昕', '女', '1990-4-23', '1898888888', NULL, 2);

SELECT * FROM beauty;


# 方式2
INSERT INTO beauty
SET id=19, NAME='刘涛', phone='999';

SELECT * FROM beauty;


# 二、修改
# 单表修改
# 案例1：修改beauty表中姓唐的女神电话为13899888899
UPDATE
  beauty
SET
  phone = '13899888899'
WHERE NAME LIKE '唐%';

# 案例2：修改2号鹿晗为公孙衍，魅力值改为10
UPDATE boys SET boyName='张飞', userCP=10
WHERE id=2;



SELECT * FROM boys;


# 多表修改
 # 案例2：修改张无忌的女朋收的手机号为114
UPDATE
  boys bo
  INNER JOIN beauty b
    ON b.`boyfriend_id` = bo.`id` SET b.`photo` = '114'
WHERE bo.`boyName` = '张无忌';

SELECT * FROM  beauty;

# 案例3：修改没有男朋友的女神的男朋友为id

UPDATE
  beauty b
  LEFT JOIN boys bo
    ON b.`boyfriend_id` = bo.`id` 
  SET b.`boyfriend_id` =
    (SELECT
      id
    FROM
      boys
    WHERE boyName = '张飞')
WHERE bo.`id` IS NULL;

SELECT * FROM beauty;


# 删除

# 方式一、 delete

# 1. 单表的删除
# 案例1：删除手机编号以9结尾的女神信息
DELETE FROM beauty WHERE phone LIKE '%9';
SELECT * FROM beauty;

# 2. 多表的删除
# 案例1：删除张无忌的女朋友的信息
DELETE
  b
FROM
  beauty b
  INNER JOIN boys bo
    ON b.`boyfriend_id` = bo.`id`
WHERE bo.`boyName` = '张无忌';

SELECT * FROM beauty;

# 案例2：删除黄晓明的信息以及他女朋友的信息
DELETE
  bo,
  b
FROM
  beauty b
  INNER JOIN boys bo
    ON b.`boyfriend_id` = bo.`id`
WHERE bo.`boyName` = '黄晓明';

SELECT * FROM beauty;
SELECT * FROM boys;

# 方式二、tuncate语句
TRUNCATE TABLE boys;


# PK
SELECT * FROM boys;

DELETE FROM boys;
INSERT INTO boys (boyName, userCP) 
VALUE ('刘备', 100), ('关羽', 100), ('张飞', 100);

TRUNCATE TABLE boys;

INSERT INTO boys (boyName, userCP)
VALUE ('刘备', 100),('关羽', 100),('张飞', 100);



# 测试题

# 1. 运行以下脚本创建表my_employees

USE myemployees;
CREATE TABLE my_employees(
	Id INT(10),
	First_name VARCHAR(10),
	Last_name VARCHAR(10),
	Userid VARCHAR(10),
	Salary DOUBLE(10,2)
);
CREATE TABLE users(
	id INT,
	userid VARCHAR(10),
	department_id INT

);

# 2. 显示my_employees的表结构
DESC my_employees;

# 3.	向my_employees表中插入下列数据
ID	FIRST_NAME	LAST_NAME	USERID	SALARY
1	patel		Ralph		Rpatel	895
2	Dancs		Betty		Bdancs	860
3	Biri		Ben		Bbiri	1100
4	Newman		Chad		Cnewman	750
5	Ropeburn	Audrey		Aropebur  1550

# 方式1
INSERT INTO my_employees (ID,FIRST_NAME,LAST_NAME,USERID,SALARY)
VALUE (1,'patel','Ralph','Rpatel',895),
(2,'Dancs','Betty','Bdancs',860),
(3,'Biri','Ben','Bbiri',1100),
(4,'Newman','Chad','Cnewman',750),
(5,'Ropeburn','Audrey','Aropebur',1550);

TRUNCATE TABLE my_employees;

INSERT INTO my_employees
SELECT 1,'patel','Ralph','Rpatel',895 UNION
SELECT 2,'Dancs','Betty','Bdancs',860 UNION
SELECT 3,'Biri','Ben','Bbiri',1100 UNION
SELECT 4,'Newman','Chad','Cnewman',750 UNION
SELECT 5,'Ropeburn','Audrey','Aropebur',1550;

SELECT * FROM my_employees;

# 4. 向users表中插入数据
1	Rpatel	10
2	Bdancs	10
3	Bbiri	20
4	Cnewman	30
5	Aropebur	40

INSERT INTO users
VALUE (1,'Rpatel',10),
(2,'Bdancs',10),
(3,'Bbiri',20),
(4,'Cnewman',30),
(5,'Aropebur',40);

SELECT * FROM users;

#5.将3号员工的last_name修改为“drelxer”

UPDATE my_employees
SET Last_name='drelxer'
WHERE id=3;

#6.将所有工资少于900的员工的工资修改为1000
UPDATE my_employees
SET Salary=1000
WHERE Salary<900;

SELECT * FROM my_employees;

#7.将userid 为Bbiri的user表和my_employees表的记录全部删除
DELETE u, e 
FROM users u
INNER JOIN my_employees e ON u.`userid`=e.`Userid`
WHERE u.`userid`='Bbiri';

#8.删除my_employees所有数据

DELETE FROM my_employees;

# 9. 清空users
TRUNCATE TABLE users;









# DDL语言

# 一、库的管理

# 1. 库的创建


# 案例：创建库books
CREATE DATABASE books;

# 2、修改库

# 修改库的字符集
ALTER DATABASE books CHARACTER SET gbk;

# 3、删除库
DROP DATABASE IF EXISTS books;


# 二、表的管理

# 1. 表的创建 ★

# 案例：再books下创建一个表book
CREATE TABLE IF NOT EXISTS book(
id INT,  # 编号
book_name VARCHAR(20),  # 图书名，20指的是书名的最大长度 
author_id INT,  # 使用作者id而非作者name，防止冗余
price DOUBLE,  # 价格
publish_date DATETIME  # 出版日期
);

DESC book;

# 案例：创建作者表

CREATE TABLE author(
id INT,
author_name VARCHAR(20),
nation VARCHAR(20)
);

DESC author;

# 2、表的修改

# ① 修改列名
ALTER TABLE book CHANGE COLUMN publish_date pub_date DATETIME;

# ② 修改列的类型
ALTER TABLE book MODIFY COLUMN pub_date TIMESTAMP;
DESC book;

# ③ 添加新列
ALTER TABLE author ADD COLUMN annual DOUBLE;
DESC author;

# ④ 删除列
# 重复删除会报错
ALTER TABLE author DROP COLUMN category;


# ⑤ 修改表名
ALTER TABLE author RENAME TO book_author;

# 3、表的删除

DROP TABLE book_author;

SHOW TABLES;

# 4、表的复制

# 先插入
INSERT INTO author 
VALUE (1, '村上春树', '日本'),
(2, '莫言', '中国');
SELECT * FROM author;

# 1. 仅仅复制表的结构

CREATE TABLE copy_author LIKE author;
DESC copy_author;
SELECT * FROM copy_author;

# 仅仅复制表的部分结构，即部分字段
CREATE TABLE copy5 LIKE  # 错误，这种情况下不能使用like
(SELECT author_name, nation
FROM author);

CREATE TABLE copy4 
SELECT author_name, nation
FROM author
WHERE 0;  # o代表false，则此时不会复制数据
SELECT * FROM copy4;

# 2. 复制表的结构+数据
CREATE TABLE copy2_author 
SELECT * FROM author;
SELECT * FROM copy2_author;

# 只复制部分数据（行或列），一切由select字句决定
CREATE TABLE copy2
SELECT author_name, nation
FROM author
WHERE nation='中国';

SELECT * FROM copy2;



# 测试
USE test;
# 1. 创建表 dept1

CREATE TABLE dept1(
id INT(7),
NAME VARCHAR(25)
);

DESC dept1;

# 2. 将departments的数据插入到新表dept2中
CREATE TABLE dept2
SELECT * FROM myemployees.`departments`;


# 3. 创建表 emp5

CREATE TABLE emp5(
id INT(7),
First_name VARCHAR (25), 
Last_name VARCHAR(25),
Dept_id INT(7)
);

# 4. 将列 Last_name 的长度增加到 50
ALTER TABLE emp5 MODIFY Last_name VARCHAR(50);

# 5. 根据表 employees 创建 employees2
CREATE TABLE employees2 LIKE myemployees.`employees`;
# 6. 删除表 emp5
DROP TABLE IF EXISTS emp5;
# 7. 将表 employees2 重命名为 emp5;
ALTER TABLE employees2 RENAME TO emp5;
# 8 在表 dept 和 emp5 中添加新列 test_column，并检查所作的操作
ALTER TABLE emp5 ADD COLUMN test_column INT(1);
# 9.直接删除表 emp5 中的列 dept_id
DESC emp5;
ALTER TABLE emp5 DROP COLUMN test_column;







# 常见的数据类型

# 1. 整型

# 1. 如何设置有符号整型和无符号整型
CREATE TABLE tab_int(
t1 INT,
);

INSERT INTO tab_int VALUE(-2222);

ALTER TABLE tab_int ADD COLUMN t2 INT UNSIGNED;
DESC tab_int;

INSERT INTO tab_int VALUE(-123, -2222);  # 无符号证书类型插入负数会报错
SELECT *FROM tab_int;


# 2. 浮点型

DROP TABLE tab_float;
CREATE TABLE tab_float(
f1 FLOAT(5, 2), 
f2 DOUBLE(5, 2),
f3 DECIMAL(5, 2)
);
SELECT * FROM tab_float;
INSERT INTO tab_float VALUE(123.45, 123.45, 123.45);
INSERT INTO tab_float VALUE(123.456, 123.456, 123.456);
INSERT INTO tab_float VALUE(123.4, 123.4, 123.4);
INSERT INTO tab_float VALUE(1123.456, 1123.456, 1123.456);

CREATE TABLE tab_float(
f1 FLOAT, 
f2 DOUBLE,
f3 DECIMAL
);
INSERT INTO tab_float VALUE(1123.456, 1123.456, 1123.456);
DESC tab_float;




# 3. 字符型
CREATE TABLE tab_char(
c1 ENUM('a', 'b', 'c')
);
SELECT * FROM tab_char;
INSERT INTO tab_char VALUE('a');
INSERT INTO tab_char VALUE('b');
INSERT INTO tab_char VALUE('d');
INSERT INTO tab_char VALUE('A');
INSERT INTO tab_char VALUE('B');


CREATE TABLE tab_set(
s1 SET('a', 'b', 'c'));
INSERT INTO tab_set VALUES('a');
INSERT INTO tab_set VALUES ('a,b');

SELECT * FROM tab_set;



CREATE TABLE tab_date(
t1 DATETIME,
t2 TIMESTAMP
);

INSERT INTO tab_date VALUES (NOW(), NOW()); 
SELECT * FROM tab_date;
SHOW VARIABLES LIKE 'time_zone';
SET time_zone='+9:00';




# 约束

# 一. 创建表时添加约束
CREATE DATABASE students;
USE students;

CREATE TABLE major(
	id INT PRIMARY KEY,
	major_name VARCHAR(20)
);

DROP TABLE IF EXISTS stuinfo;

CREATE TABLE IF NOT EXISTS stuinfo(
	id INT PRIMARY KEY,
	stu_name VARCHAR(20) UNIQUE,
	age INT DEFAULT 18,
	seat INT UNIQUE,
	major_id INT,
	# 表级约束
	CONSTRAINT fk_stuinfo_major FOREIGN KEY(major_id) REFERENCES major(id)
);

SHOW INDEX FROM stuinfo;

DESC stuinfo;

# 二、修改表是添加约束
ALTER TABLE stuinfo MODIFY COLUMN stu_name VARCHAR(20) NOT NULL;

ALTER TABLE stuinfo MODIFY COLUMN age INT DEFAULT 20;
DESC stuinfo;


# test
USE test;

# 向emp5的employee_id 添加主键约束
DESC emp5;
SHOW INDEX FROM emp5;
ALTER TABLE emp5 DROP PRIMARY KEY;
ALTER TABLE emp5 MODIFY COLUMN employee_id INT PRIMARY KEY;

# 向dept2的department_id列添加主键约束(my_dept_id_pk)
DESC dept2;
ALTER TABLE dept2 ADD CONSTRAINT my_dept_id_pk PRIMARY KEY(department_id);

# 向emp2中添加列dept_id，并在其中定义外键约束，与之相关联的是dept2中的department_id列
SHOW INDEX FROM emp5;
# 先添加列再修改约束
ALTER TABLE emp5 ADD COLUMN dept_id INT;
ALTER TABLE emp5 ADD CONSTRAINT fk_emp2_dept2 FOREIGN KEY(dept_id) REFERENCES dept2(depatment_id);




# 标识列

CREATE TABLE tab_indentify(
	id INT PRIMARY KEY AUTO_INCREMENT,
	NAME VARCHAR(20)
);
DROP TABLE tab_indentify;
SELECT * FROM tab_indentify;
TRUNCATE TABLE tab_indentify;

INSERT INTO tab_indentify VALUES(NULL, 'john');  # 为了保证和字段列表一致，插入null


SHOW VARIABLES LIKE '%auto_increment%';
SET auto_increment_increment=1;
SET auto_increment_offset=1;
INSERT INTO tab_indentify VALUES(NULL, 'mike');









# TCL语言

#存储引擎
SHOW ENGINES;

SHOW VARIABLES LIKE 'autocommit';  # 自动提交on


# 演示事务的使用步骤

DROP TABLE IF EXISTS account;
CREATE TABLE account(
	id INT PRIMARY KEY AUTO_INCREMENT,
	username VARCHAR(20),
	balance DOUBLE
);
SELECT * FROM account;

INSERT INTO account (username, balance) VALUES ('张无忌', 1000), ('赵敏', 1000);

# 步骤1：开启事务
SET autocommit=0;
START TRANSACTION;

# 步骤2：编写一组事务的语句
UPDATE account SET balance = 1000 WHERE username = '张无忌';
UPDATE account SET balance = 1000 WHERE username = '赵敏';

# 步骤3：结束事务
# 提交
COMMIT;  # 只有commit才会提交

# 回滚
ROLLBACK;

# 2. delete 和 truncate在事务使用中的区别
SET autocommit=0;
START TRANSACTION;

DELETE FROM account;
ROLLBACK;

SET autocommit;
START TRANSACTION;

TRUNCATE TABLE account;
ROLLBACK;

# savapoint
SELECT * FROM account;
SET autocommit=0;
START TRANSACTION;
DELETE FROM account WHERE id=3;
SAVEPOINT a;
DELETE FROM account WHERE id=1;
ROLLBACK TO a;



















# 视图
USE myemployees;

# 案例1：查询姓名中包含a字符的员工名、部门名和工种信息

# 创建视图
DROP VIEW v1;
CREATE VIEW v1 AS
SELECT
  e.`last_name`,
  d.`department_name`,
  j.`job_title`
FROM
  employees e
  INNER JOIN departments d
    ON e.`department_id` = d.`department_id`
  INNER JOIN jobs j
    ON e.`job_id` = j.`job_id`;

# 使用
SELECT * FROM v1 WHERE last_name LIKE '%a%';



# 测试

# 1. 创建视图，要求查询电话号码以'011'开头的员工名和工资、邮箱
DESC employees;
DROP VIEW v2;
CREATE OR REPLACE VIEW v2 AS
SELECT
  last_name,
  salary,
  email,
  phone_number
FROM
  employees;

SELECT * FROM v2 
WHERE phone_number LIKE '011%';

# 2. 创建视图emp_v2，要求查询部门的最高工资高于12000的部门信息
CREATE OR REPLACE VIEW emp_v2 AS
SELECT
  MAX(salary) max_salary,
  d.*
FROM
  employees e
  INNER JOIN departments d
    ON d.`department_id` = e.`department_id`
GROUP BY e.`department_id`;

SELECT * FROM emp_v2 WHERE max_salary>12000;


# 视图的更新
DROP VIEW v1, v2, emp_v2;

CREATE OR REPLACE VIEW v1
AS 
SELECT last_name, email FROM employees;

SELECT * FROM v1;
SELECT * FROM employees WHERE last_name='张飞';
SELECT * FROM employees WHERE last_name='张无忌';

# 1. 插入
INSERT INTO v1 VALUES('张飞', 'zf@qq.com');
# 视图中插入新的数据时，会同时对原始表进行更新。
# 2. 修改
UPDATE v1 SET last_name='张无忌' WHERE last_name='张飞';
# 3. delete
DELETE FROM v1 WHERE last_name='张无忌';





# 测试

# 1. 创建一个book表，字段如下：
bid 整型，要习
bname 字符型，要求设置唯一键，并非空
price 浮点型，要求有默认值10
btypeId类型编号，要求引用bookType表的id字段;

CREATE TABLE book(
	bid INT PRIMARY KEY,
	bname VARCHAR(20) NOT NULL UNIQUE,
	price FLOAT DEFAULT 10,
	btypeId INT,
	FOREIGN KEY(btypeId) REFERENCES bookType(id)
);



已知bookType表,字段如下：
id
NAME;
CREATE TABLE bookType(
	id INT PRIMARY KEY,
	book_name VARCHAR(20)
);
# 2、开启事务, 向表中插入1行数据，并结束
# insert数据先insert主表，在insert从表
SET autocommit=0;
DESC bookType;
INSERT INTO bookType VALUE(2, '历史');

INSERT INTO book
VALUES(1, '战国策', 30, 2);
COMMIT;
# 3、创建视图，实现查询价格大于100的书名和类型名
CREATE OR REPLACE VIEW v1 AS
SELECT
  b.`bname`,
  b.`price`, 
  bt.`book_name`
FROM
  book b
  INNER JOIN bookType bt
    ON b.`btypeId` = bt.`id`;
SELECT * FROM v1 WHERE price>100;


# 4 、删除刚才建的视图
DROP VIEW v1;



















# 变量

# 全局变量的使用
# 查看所有的全局系统变量
SHOW GLOBAL VARIABLES;
# 查看部分全局变量
SHOW GLOBAL VARIABLES LIKE '%char%';
# 查看指定的全局变量
SELECT @@global.autocommit;
SELECT @@global.tx_isolation;
# 为指定的全局变量赋值
SET @@global.autocommit=0;
SET GLOBAL autocommit=1;

# 会话变量
# 查看所有的会话变量
SHOW SESSION VARIABLES;

# 自定义变量

# 用户变量

# 案例：声明两个变量并赋初值，求和并打印
# 用户变量
SET @m=1;
SELECT @n:=2;
SET @sum=@m+@n;
SELECT @sum;

# 局部变量
BEGIN;
DECLARE m INT DEFAULT 1;
DECLARE n INT DEFAULT 2;
DECLARE SUM INT;
SET SUM=m+n;
SELECT SUM;

END;















# 存储过程和函数

# 1. 空参列表

# 案例：插入到admin表中五条记录
USE girls;
SELECT * FROM admin;

DELIMITER ;
CREATE OR REPLACE PROCEDURE mypl()
BEGIN
	INSERT INTO admin(username, `password`)
	VALUES('john', '0000'), ('lily', '0000'), ('rose', '0000'), ('jack', '0000'), ('tom', '0000');
END ;

# 调用
CALL mysql()$

# 2. 创建带in模式参数的存储过程
# 案例1：创建存储过程实现根据女神名，查询对应的男神信息

DELIMITER /
CREATE PROCEDURE myp2(IN beauty_name VARCHAR(20))
BEGIN
	SELECT bo.*
	FROM boys bo
	RIGHT JOIN beauty b ON bo.id=b.boyfriend_id
	WHERE b.name = beauty_name;
END /

# 调用
CALL myp2('柳岩')/

# 案例2. 创建存储过程实现，用户是否登陆成功

CREATE PROCEDURE myp4(IN username VARCHAR(20), IN PASSWORD VARCHAR(20))
BEGIN 
	DECLARE result INT DEFAULT 0;  # 变量的声明并初始化
	SELECT COUNT(*) INTO result  # 变量的赋值
	FROM admin
	WHERE admin.`username`=username
	AND admin.`password`=PASSWORD; 
	
	SELECT IF(result>0, '成功', '失败');  # 变量的使用
END /

CALL myp4('张飞', '8888') /

# 3. 创建带out模式的存储

# 案例1：根据女神名返回对应的男神名
DESC beauty;
DESC boys;
CREATE PROCEDURE myp5(IN beauty_name VARCHAR(20), OUT boy_name VARCHAR(20))
BEGIN 
	SELECT bo.boyName INTO boy_name
	FROM boys bo
	RIGHT JOIN beauty b ON bo.id=b.boyfriend_id
	WHERE b.name=beauty_name;
END /

# 调用
# 方法一：外部定义一个用户变量，让其接收out参数
SET @boy_name=1/  # 可选，用户变量定义的同时必须赋初值
CALL myp5('柳岩', @boy_name)/

# 方法二：直接在调用时的实参列表中定义用户变量
CALL myp5('柳岩', @boy_name)/

SELECT @boy_name/

# 案例2：根据女神名，返回对应的男神名和魅力值
CREATE PROCEDURE myp6(IN beauty_name VARCHAR(20), OUT boy_name VARCHAR(20), OUT 魅力值 INT)
BEGIN 
	SELECT bo.boyName, bo.userCP INTO boy_name, 魅力值
	FROM boys bo
	RIGHT JOIN beauty b ON b.boyfriend_id=bo.id
	WHERE b.name=beauty_name;
END /

# 调用
CALL myp6('柳岩', @男, @魅力值)/

SELECT @男, @魅力值/


# 4. 创建带inout模式参数的存储过程
# 案例1：闯入a和b两个值，最终a和b都返回double
CREATE PROCEDURE myp8(INOUT a INT, INOUT b INT)
BEGIN 
	SET a=a*2;
	SET b=b*2;
END /

# 调用
# 先定义两个用户变量
SET @m=10;
SELECT @n:=20;
CALL myp8(@m, @n);
SELECT @m, @n/


# 测试
CREATE PROCEDURE test_pro1(IN username VARCHAR(20), IN log_password VARCHAR(20))
BEGIN 
	INSERT INTO admin(admin.`username`, PASSWORD)
	VALUES(username, log_password);
END /


# 删除存储过程
DROP PROCEDURE mypl;

# 查看存储过程
SHOW CREATE PROCEDURE myp2;








# 测试
# 案例：创建存储过程或函数实现传入一个日期，格式化成xx年xx月xx日并返回

CREATE PROCEDURE myp7(IN 日期 DATE, OUT xx VARCHAR(20))
BEGIN 
	SELECT DATE_FORMAT(日期, '%Y年%c月%d日') INTO xx;
END /

CALL myp7('1995-12-26', @日期)/
SELECT @日期/



# 函数

# 创建
DELIMITER /
# ------------------案例--------------------------
# 1.无参有返回
# 案例：返回公司的员工个数
CREATE FUNCTION myf1() RETURNS INT
BEGIN
	DECLARE c INT DEFAULT 0;
	SELECT COUNT(*) INTO c FROM employees;
	RETURN c;
END /

# 调用
SELECT myf1()/

# 有参又返回
# 案例1：根据员工名返回工资
CREATE FUNCTION myf2(员工名 VARCHAR(20)) RETURNS DOUBLE
BEGIN
	SET @sal=0;  # 定义用户变量
	SELECT salary INTO @sal FROM employees WHERE last_name=员工名;
	RETURN @sal;
END /

SELECT myf2('Ernst')/

# 案例2：根据部门名返回该部门的平均工资
CREATE FUNCTION myf3(部门名 VARCHAR(20)) RETURNS DOUBLE
BEGIN
	DECLARE sal DOUBLE;
	SELECT AVG(salary) INTO sal
	FROM employees e
	INNER JOIN departments d ON e.`department_id`=d.`department_id`
	WHERE 部门名=d.`department_name`;
	# group by department_id; 不需要分组，因为在where里面已经选出了该部门
	RETURN sal;
END /

SELECT myf3('IT')/

# 创建函数：要求传入两个float，返回二者之和
USE test;
CREATE FUNCTION myf4(a FLOAT, b FLOAT) RETURNS FLOAT
BEGIN
	DECLARE two_sum FLOAT DEFAULT 0;
	SELECT a+b INTO two_sum;
	RETURN two_sum;
END/



# 查看函数
SHOW CREATE FUNCTION myf3;

# 删除函数
DROP FUNCTION myf3;


































#  流程控制结构

# ----------------IF函数

# ----------------case语句

# 案例
DELIMITER /
# 创建存储过程，根据传入的成绩，显示等级，例如：成绩在90-100，显示A，80-89，显示B，60-79，显示C，否则显示D
CREATE PROCEDURE test_case(IN grade INT)
BEGIN
	DECLARE class CHAR;
	CASE
	WHEN grade>=90 THEN SET class='A';
	WHEN grade>=80 THEN SET class='B';
	WHEN grade>=60 THEN SET class='C';
	ELSE SET class='D';
	END CASE;
	SELECT class;
END /

CALL test_case(94) /

# ----------------IF分支结构
# 位置：应用在begin
# 案例1：根据传入的成绩，显示等级，例如：成绩在90-100，返回A，80-89，返回B，60-79，返回C，否则返回D
DELIMITER /
CREATE FUNCTION test_if(score INT) RETURNS CHAR
BEGIN
	IF score BETWEEN 90 AND 100 THEN RETURN 'A';
	ELSEIF score BETWEEN 80 AND 89 THEN RETURN 'B';
	ELSEIF score BETWEEN 60 AND 79 THEN RETURN 'C';
	ELSE RETURN 'D';
	END IF;
END /
SELECT test_if(86)/













# 循环结构

# 案例：批量插入，根据次数插入到admin表中多条记录
DROP PROCEDURE pro_while;
CREATE PROCEDURE pro_while(IN insertcount INT)
BEGIN
	DECLARE i INT DEFAULT 1;
	WHILE i<=insertcount DO
		INSERT INTO admin(`username`, `password`) VALUES(CONCAT('Rose', i), '666');
		SET i=i+1;
	END WHILE;
END /

CALL pro_while(100)/

# 添加leave语句
# 案例：批量插入，根据次数插入到admin表中多条记录, 最多插入20次
CREATE PROCEDURE pro_while2(IN insertcount INT)
BEGIN
	DECLARE i INT DEFAULT 1;
	a: WHILE i<=insertcount DO
		INSERT INTO admin(`username`, `password`) VALUES(CONCAT('jack', i), '666');
		SET i=i+1;
		IF i>20 THEN LEAVE a;
		END IF;
	END WHILE a;
END /

CALL test_while2(50)/

# 添加iterate语句
# 只插入偶数次
CREATE PROCEDURE pro_while3(IN insertcount INT)
BEGIN
	DECLARE i INT DEFAULT 0;
	a: WHILE i<=insertcount DO
		SET i=i+1;
		IF i%2=1 THEN ITERATE a;
		END IF;
		INSERT INTO admin(`username`, `password`) VALUES(CONCAT('mike', i), '666');
		IF i>20 THEN LEAVE a;
		END IF;
	END WHILE a;
END /







# 测试：
/*
已知表stringcontent
其中字段：
id 自增长
content varchar(20)
向该表插入指定个数的，随机的字符串
*/

CREATE TABLE stringcontent(
	id INT PRIMARY KEY AUTO_INCREMENT,
	content VARCHAR(20)
);

CREATE PROCEDURE random_str(IN times INT)
BEGIN
	DECLARE i INT DEFAULT 1;
	DECLARE j INT DEFAULT 1;
	DECLARE str VARCHAR(26) DEFAULT 'abcdefghigklmnopqrstuvwxyz';
	DECLARE startindex INT DEFAULT 1;  # 起始索引
	DECLARE str_length INT DEFAULT 1;  # 字符串的长度
	DECLARE single_str CHAR;
	DECLARE mid_str VARCHAR(20);
	a: WHILE i<=times DO
		# 获取一个随机的字符串
		SET str_length=CEIL(RAND()*20);
		b: WHILE j<=str_length DO
			SET startindex=CEIL(RAND()*26); # 产生一个随机的字符
			SET single_str=SUBSTR(str, startindex, 1);
			IF j=1 THEN 
				SET mid_str=single_str;
			ELSE
				SET mid_str=CONCAT(mid_str, single_str);
			END IF;
			SET j=j+1;
		END WHILE b;
		INSERT INTO stringcontent(`content`) VALUES(mid_str);
		SET i=i+1;  # 循环变量自增
	END WHILE a;
END/


DROP PROCEDURE random_str;
CALL random_str(1)/
SELECT *FROM stringcontent/

SELECT CEIL(RAND()*26);
SELECT RAND(10);
SELECT CONCAT(NULL, 'a');