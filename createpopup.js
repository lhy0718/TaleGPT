function createPopupWithText() {
  const imagePath = window.location.href + "file=images/paper.jpg"
  // 특정 요소에서 텍스트 가져오기
  const titleElement = document.querySelectorAll('[data-testid="user"]')[0]; // 대상 요소의 ID를 지정하세요
  const title = titleElement.innerText;
  const contentElement = document.querySelectorAll('[data-testid="bot"]')[0]; // 대상 요소의 ID를 지정하세요
  const content = contentElement.innerHTML

  // A4 페이지 크기 설정 (210mm x 297mm)
  const width = 210;
  const height = 297;

  // 새 팝업 창 열기
  const popup = window.open('', '_blank', `width=${width}mm,height=${height}mm`);

  // 팝업 내부에 HTML을 작성
  const popupContent = `
    <!DOCTYPE html>
    <html>
    <head>
      <title>A4 출력</title>
      <style>
        @import url('https://fonts.googleapis.com/css2?family=Diphylleia&display=swap');
        @page {
          size: A4;
          margin: 0;
        }
        body {
          width: ${width}mm;
          height: ${height}mm;
          margin: 0;
          font-size: 20pt;
          padding: 50px;
          font-family: 'Diphylleia', serif;
          background-image: url('${imagePath}');
          background-size: ${width/20*21}mm ${height/20*21}mm;
          background-repeat: no-repeat;
          background-attachment: fixed;
        }
      </style>
    </head>
    <body>
      <h1>${title}</h1>
      <div>${content}</div>
    </body>
    </html>
  `;

  // 팝업에 내용 삽입
  popup.document.open();
  popup.document.write(popupContent);
  popup.document.close();
}

// 팝업 생성
createPopupWithText();
