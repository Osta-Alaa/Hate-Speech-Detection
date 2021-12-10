$(document).ready(function() {
    console.log("Hello");
    $('.loader').hide();
    $('.sentiment').hide();
    // Make prediction by calling api /predict
    $('#btn-predict').click(function() {
        var text = new FormData($('#text-form')[0]);
        console.log(text);
        // Show loading animation
        $(this).hide();
        $('.sentiment').hide();
        $('.loader').show();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: text,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function(data) {
                // Get and display the result
                $("#text").text(data);
                $('.loader').hide();
                $('#btn-predict').show();

                console.log('Success!');
                if (data == "1") {
                    $('.sentiment h2').text("😡 speech is detected");

                } else if (data == "0") {
                    $('.sentiment h2').text("😃 speech is detected");
                }
                $('.sentiment').fadeIn(600);
            }

        }, );
    });
});