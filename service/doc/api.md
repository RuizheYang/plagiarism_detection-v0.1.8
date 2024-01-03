查重

POST /api/duplicate/check

Request body (json):

```
{
	title: “标题”,
	article: [
		“段落”
	],
	files: [
	    {   id: 123
	        name: "文件名"(也就是标题)
	        content: "正文"
	        isUserFile: true// true - 是用户文件，false - 政策大脑文件
			url: ‘’// 政策大脑的文件有连接
	    },
	    ...
	]
	ignore: [
		“忽略的文本”
	]
}
```
Response:
```
{
	percent: “90%”, // 综合重复率
	fileList: [
        {
			id: 123,
			name: ’文件名’,
			percent: ’50%’,
			isUserFile: true // true - 是用户文件，false - 政策大脑文件
			url: ‘’ // 政策大脑的文件有连接
        }
    ],
	title: {} // 同下面的article的一项
	article: [
        {
			text: “段落”,
			result: [
                {
					pos: 1,
					len: 10,
					files: [
                        {
							name: “中共中央。。。”,
							percent: “50%”,
							preview: “这是重复内容部分”
                        }
                    ]
                }
            ]
        }
    ]
}
```