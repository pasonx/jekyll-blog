---
published: true
---
本文介绍Spring Boot中注解对URL的解析方法

## @RequestMapping
此注解用于将URL某一模式或路径映射到具体的处理方法上面
{% highlight java %}
	 //path后面可以有多个模式用逗号分隔，它们都映射到同一个处理函数
    //method定义了http请求的方式，当定义为GET时其他方式不被允许
    @RequestMapping(path={"/","/index"}, method = {RequestMethod.GET})
    @ResponseBody
    public String index() {
        return "Hello,Spring Boot!";
    }
{% endhighlight %}
