// This is the scripts file for the web interface for my Machine Learning Model.

// This function is called when the page is loaded.
function init() { 
    // Get the current URL.
    var url = window.location.href;
    // Get the current page.
    var page = url.split('/').pop();
    // Get the current page name.
    var page_name = page.split('.')[0];
    // Get the current page element.
    var page_element = document.getElementById(page_name);
    // Add the active class to the current page element.
    page_element.classList.add('active');
}

// This function is called when a user clicks the browse button to upload a file.
function browse() {
    // Get the file input element.
    var file_input = document.getElementById('file_input');
    // Click the file input element.
    file_input.click();
}

// This function is called when a user selects a file to upload.
function file_selected() {
    // Get the file input element.
    var file_input = document.getElementById('file_input');
    // Get the file name element.
    var file_name = document.getElementById('file_name');
    // Set the file name element to the name of the file.
    file_name.innerHTML = file_input.files[0].name;
}

// This function is called when a user clicks the upload button.
function upload() { 
    // Get the file input element.
    var file_input = document.getElementById('file_input');
    // Get the file name element.
    var file_name = document.getElementById('file_name');
    // Get the file name.
    var file = file_input.files[0].name;
    // Get the file extension.
    var extension = file.split('.').pop();
    // Check if the file extension is csv.
    if (extension == 'jpg' || extension == 'jpeg' ) {
        // Get the form element.
        var form = document.getElementById('form');
        // Submit the form.
        form.submit();
    }
    else {
        // Get the error message element.
        var error_message = document.getElementById('error_message');
        // Set the error message element to the error message.
        error_message.innerHTML = 'Please upload a jpg or jpeg.';
        // Get the error message element.
        error_message.style.display = 'block';
    }
}