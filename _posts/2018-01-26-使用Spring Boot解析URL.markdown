---
published: false
---
---
layout: post
title:  "image augmentation processing"
date:   2018-01-26 17:02:13 +0000
categories: 
---

本文介绍Spring Boot中注解对URL的解析方法

## @RequestMapping
此注解用于将URL某一模式或路径映射到具体的处理方法上面
```
    @RequestMapping(path={"/","/index"})
    @ResponseBody
    public String index() {
        return "Hello,Tensorflow";
    }
```
