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

## 处理URL中的参数
   在URL中，我们要先取得URL中每一个/item/的内容，可以现在RequestMapping里面写成/profile/{groupId}/{userId}的形式，花括号里我们先定义一个占位符变量，在之后的处理函数的参数中使用@PathVariable注解便可以将每一个/item/取出来。
    另外，在URL后面仍然有许多重要的参数传递，往往以{% highlight java %}?type=xxx&key=xxx{% endhighlight %}这样的形式出现，同理我们使用@RequestParam来处理
{% highlight java %}

    @RequestMapping(value = {"/profile/{groupId}/{userId}"}, method = {RequestMethod.GET})
    @ResponseBody
    public String profile(@PathVariable("groupId")String groupId,
                   @PathVariable("userId")int userId,
                   @RequestParam(value = "type", defaultValue = "1") int type,
                   @RequestParam(value = "key", required = false) String key) {

        return String.format("Gourp id is %s, user id is %d, type is %d, key is %s}", groupId, userId, type, key);
    }
    
{% endhighlight %}
