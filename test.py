l = iter([1,2,3])

def function(iterable):
	sentinel = object()
	print(next(iterable))

function(l)
function(l)
function(l)
function(l)
function(l)