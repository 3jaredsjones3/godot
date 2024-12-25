#include "vector4.h"
#include "core/string/ustring.h"  // For String class
#include "vector4i.h"            // For Vector4i class


Vector4::operator String() const {
    return String("{x: ") + String::num(x) + ", y: " + String::num(y) +
           ", z: " + String::num(z) + ", w: " + String::num(w) + "}";
}

Vector4::operator Vector4i() const {
    return Vector4i(int(x), int(y), int(z), int(w));
}
