package util

import (
	"strings"

	"github.com/google/uuid"
)

func GetUUID() string {
	code := uuid.New().String()
	code = strings.ReplaceAll(code, "-", "")
	return code
}
