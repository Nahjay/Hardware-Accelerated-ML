// This function is called when the page is loaded.
function init() {
    // Get the current URL.
    var url = window.location.href;
    // Get the index of the last slash.
    var index = url.lastIndexOf('/');
    // Get the last part of the URL.
    var page = url.substring(index + 1);

    // Check if the page variable is empty.
    if (page === '') {
        // Set the page variable to index.
        page = 'index';
    }

    // Set the active class to the current page element.
    var page_element = document.getElementById(page);
    if (page_element) {
        page_element.classList.add('active');
    }
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

    // Check if the file input element is empty.
    if (file_input.files.length === 0) {
        // Get the error message element.
        var error_message = document.getElementById('error_message');
        // Set the error message element to the error message.
        error_message.innerHTML = 'Please select a file.';
        // Show the error message element.
        error_message.style.display = 'block';
        return;
    }

    // Get the file name.
    var file = file_input.files[0].name;

    // Get the file extension.
    var extension = file.split('.').pop();
    // Check if the file extension is jpg or jpeg.
    if (extension.toLowerCase() === 'jpg' || extension.toLowerCase() === 'jpeg') {
        // Get the form element.
        var form = document.getElementById('form');
        // Submit the form.
        form.submit();
    } else {
        // Get the error message element.
        var error_message = document.getElementById('error_message');
        // Set the error message element to the error message.
        error_message.innerHTML = 'Please upload a jpg or jpeg.';
        // Show the error message element.
        error_message.style.display = 'block';
    }
}


// Call the init function when the page is loaded.
document.addEventListener('DOMContentLoaded', init);
document.getElementById('file_input').addEventListener('change', file_selected);
document.getElementById('upload_button').addEventListener('click', upload);
