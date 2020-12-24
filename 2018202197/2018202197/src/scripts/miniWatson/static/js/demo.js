var data = [
    { name: "miniWatson" },
    /*{ name: "李小小" },
    { name: "肖大大" },
    { name: "江大大" },
    { name: "黄某某" },
    { name: "陈某某" },
    { name: "苏某某" },
    { name: "陈小小" },
    { name: "刘某某" },
    { name: "黄大大" },*/
];
var html = "";
for (var i = 0; i < data.length; i++) {
    html += "<li>" + "<i class='iconfont'>&#xe752;</i>" + "<p>" + data[i].name + "</p>" + "</li>";
}
$(".chatbar-contacts-uls").html(html);
//点击按钮下拉
$(".icon").on('click', function() {
    if ($(".chatbar").is(":visible")) {
        $(".chatbar").slideUp();
        $(".icon-box").removeClass('shadow');
    } else {
        $(".chatbar").slideDown();
        $(".icon-box").addClass('shadow');
    }
});

$(".chatbar-contacts-uls li").click(function() {
    var text = $(this).find('p').text();
    $(".chatbar-messages").css({
        "transform": "translate3d(0, 0, 0)"
    });
    $('.messages-title h4').text(text);
});

$(".return-icon").click(function() {
    $(".chatbar-messages").css({
        "transform": "translate3d(100%, 0, 0)"
    });
});

//发送消息
$(".message-btn").on('click', function() {
    var message = $('.messages-content').val();
    //var name = $(".messages-title").find("h4").text();
    var name = "我";
    var messages_text = $(".messages-text");
    var timer = time();
    if (message != "undefined" && message != '') {
        var str = "<ul class='messages-text-uls'><li class='messages-text-lis'>" 
                + "<h4><i></i><span>" + name + "</span><span class='time'>"
                + timer + "</span></h4>" + "<p>" + message + "</p>"
                + "</ul></li>";
        messages_text.append(str);

        var chatbox = document.getElementsByClassName("messages-content");
        for(var i = 0;i < chatbox.length; i++){
            chatbox[i].value="";
        }

        $.ajax({
            type:"GET",
            contentType:"application/json;charset=UTF-8",
            url:"/chat/",
            data:{messages: message},
            success:function(result){
                console.log("success!")
                console.log(result)
                setTimeout(function() {
                    var str = "<ul class='messages-text-uls'><li class='messages-text-lis-w'>" 
                        + "<h4><i></i><span>" + "miniWatson" + "</span><span class='time'>"
                        + timer + "</span></h4>" + "<p>" + result.reply + "</p>"
                        + "</ul></li>";
                    messages_text.append(str);
                }, 300);
            },
            error:function(e){
                console.log(e.status);
                console.log(e.responseText)
                alert("miniWatson开小差了！")
            }
        })        
    } else {
        var messageTooltip = "<div class='message-tooltip'>不能发送空白信息</div>";
        $("body").append(messageTooltip);
        setTimeout(function() {
            $(".message-tooltip").hide();
        }, 2000);
    }
});

//时间封装
function time(type) {
    type = type || 'hh:mm'
    var timer = new Date();
    var year = timer.getFullYear();
    var month = timer.getMonth() + 1;
    var date = timer.getDate();
    var hour = timer.getHours();
    var min = timer.getMinutes();
    if (type == 'hh:mm') {
        hour = hour < 10 ? ('0' + hour) : hour;
        min = min < 10 ? ('0' + min) : min;
    }
    var time = year + "/" + month + "/" + date + "  " + hour + ":" + min;
    return time;
}

//搜索功能
$('.search-text').on('keyup', function() {
    var txt = $('.search-text').val();
    txt = txt.replace(/\s/g, '');
    $('.chatbar-contacts-uls li').each(function() {
        if (!$(this).is(':contains(' + txt + ')')) {
            $(this).hide();
        } else {
            $(this).show();
        }
    });
    return false;
});
