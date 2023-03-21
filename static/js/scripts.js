
$("form[name = signup_form]").submit(function(e){

    var $from;
    $from = $(this);
    var $error = $form.find(".error");
    let $form;
    var data =  $form.serialize();

    $.ajax({
        url: "/user/signup",
        type: "POST",
        data: data,
        dataType: "json",
        success: function(resp) {
            window.location.href="/new/";
        },
        error: function(resp){
            console.log(resp);
            $error.text(resp.responseJSON.error).removeClass("error--hidden")


        }
    });

    e.preventExtensions();
});