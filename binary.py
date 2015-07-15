import struct
import array
import sys

def to_string(input):
  output = bytearray()
  output.extend('CHAR') # add to end of binary string
  minimum_length = (len(input) + 1)
  padded_length = (((minimum_length + 3) / 4) * 4) #32bit words?
  terminator_count = (padded_length - len(input)) #how many blanks need to go to wrap to 32bit
  output.extend(struct.pack('I', padded_length)) #straight up write the number of bytes to follow
  output.extend(input) #write the input to output
  for i in range(terminator_count): #pad the output to 32bit word length
    output.extend(struct.pack('B', 0))
  return output

def to_uint32(input):
  output = bytearray()
  output.extend('UINT')
  output.extend(struct.pack('I', 4))
  output.extend(struct.pack('I', input))
  return output

def to_float32(input):
  output = bytearray()
  output.extend('FL32')
  output.extend(struct.pack('I', 4))
  output.extend(struct.pack('f', input))
  return output

def to_list(input):
  output = bytearray()
  output.extend('LIST')
  output.extend(struct.pack('I', len(input)))
  output.extend(input)
  return output

def to_dict(input):
  output = bytearray()
  output.extend('DICT')
  output.extend(struct.pack('I', len(input)))
  output.extend(input)
  return output

def to_float32_array(input):
  output = bytearray()
  output.extend('FARY')
  value_data = array.array('f', input.flatten()).tostring() #flattens to 1d array, converts to float (32bit)
  output.extend(struct.pack('I', len(value_data)))
  output.extend(value_data)
  return output

def convert_simple_dict(input):
  payload = bytearray()
  for key, value in input.items():
    payload.extend(to_string(str(key)))
    if isinstance(value, str):
      payload.extend(to_string(str(value)))
    elif isinstance(value, (int, long)):
      payload.extend(to_uint32(value))
    elif isinstance(value, float):
      payload.extend(to_float32(value))
    else:
      sys.stderr.write('Unknown type for key %s\n' % (key))
  output = to_dict(payload)
  return output

# This section is optional, implementing a binary export method for numpy arrays.
# If numpy isn't installed, it's skipped.
try:
  import numpy as n

  def numpy_array_to_binary(array):
    payload = bytearray()
    payload.extend(to_string('class'))
    payload.extend(to_string('blob'))
    payload.extend(to_string('float_bits'))
    if array.dtype == n.float32:
      payload.extend(to_uint32(32))
    elif array.dtype == n.float64:
      payload.extend(to_uint32(64))
    else:
      raise "Bad data type for blob when dumping to JSON: %s" % self.dtype
    dims_payload = bytearray()
    for i in range(len(array.shape)):
      dims_payload.extend(to_uint32(array.shape[i]))
    payload.extend(to_string('dims'))
    payload.extend(to_list(dims_payload))
    payload.extend(to_string('data'))
    payload.extend(to_float32_array(array))
    output = to_dict(payload)
    return output

except ImportError:
  pass
