import requests

option1
# r = requests.get('https://......')
r = requests.get('https://xyz.org/get?page=2&count=25')


option2:
payload = {'page':2, 'count':25}
r = requests.get('https://xyz.org/get' params=payload)


# option3: post`
payload = {'username':'henry', 'password':'testing'}
r = requests.post('https://xyz.org/get' data==payload)

# print(r.text)
print(r.json())
print(r.json()['form']) #example



print(r)

# see the content
print(dir(r))

# print the comment in text
print(r.text)

# See the response
print(r.status_code) # is it 200?
print(r.ok) # boolean
print(r.headers) # boolean


